//! 完整推荐编排。学习表、访问顺序和冻结策略始终留在原生会话中。
use crate::model::{ActionCache, Context, Key, Model, Stats};
use crate::rng::Random;
use std::collections::{HashMap, HashSet};

pub type Poll = extern "C" fn(u64, u64, u32) -> i32;

#[repr(C)]
pub struct Request {
    pub initial: Context,
    pub seed: i64,
    pub control: u64,
    pub evaluation: u64,
    pub min_visits: u64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub target: f64,
    pub learning_target: f64,
    pub has_target: u32,
    pub has_learning_target: u32,
    pub mode: u32,
    pub win: u32,
    pub train_win: u32,
    pub first_actions: [u8; 6],
}

#[repr(C)]
pub struct Report {
    pub mean: [Stats; 64],
    pub wins: [Stats; 64],
    pub actions: [u8; 64],
    pub count: u32,
    pub recommended: u32,
    pub highest_win: u32,
    pub has_wins: u32,
    pub control: u64,
    pub evaluation: u64,
    pub total_control: u64,
    pub total_trials: u64,
    pub q_states: u64,
    pub cancelled: u32,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            mean: [Stats::default(); 64],
            wins: [Stats::default(); 64],
            actions: [0; 64],
            count: 0,
            recommended: 255,
            highest_win: 255,
            has_wins: 0,
            control: 0,
            evaluation: 0,
            total_control: 0,
            total_trials: 0,
            q_states: 0,
            cancelled: 0,
        }
    }
}

pub struct Engine {
    pub model: Model,
    pub total_control: u64,
    pub total_trials: u64,
    pub hits: HashSet<Key>,
}

struct Batch {
    actions: Vec<u8>,
    stats: [Stats; 64],
    completed: u64,
    expected: u64,
}

fn seed(value: i128) -> i64 {
    i64::try_from(value).expect("sampling seed exceeds i64")
}

fn mean(s: &Stats) -> f64 {
    if s.count == 0 {
        f64::NEG_INFINITY
    } else {
        s.total / s.count as f64
    }
}
fn rate(s: &Stats) -> f64 {
    if s.target_count == 0 {
        0.0
    } else {
        s.wins as f64 / s.target_count as f64
    }
}

impl Engine {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            total_control: 0,
            total_trials: 0,
            hits: HashSet::new(),
        }
    }

    fn actions(&self, c: Context) -> Vec<u8> {
        let mut actions = [0; 64];
        let count = self.model.actions(c, &mut actions);
        actions[..count].to_vec()
    }

    pub fn choose(&self, c: Context, win: bool) -> Option<u8> {
        let actions = self.actions(c);
        if actions.is_empty() {
            return None;
        }
        self.model
            .choose(
                c,
                &actions,
                win,
                &mut [ActionCache::new(false), ActionCache::new(false)],
            )
            .into()
    }

    fn evidence(&self, c: Context, action: u8) -> bool {
        self.model.q[1]
            .get(&self.model.q_key(c, action, true))
            .is_some_and(|s| s.wins > 0)
            || (self.hits.contains(&self.model.key(c, true))
                && self.model.frozen[1].get(&self.model.key(c, true)) == Some(&action))
    }

    pub fn freeze(&mut self, c: Context, action: u8, win: bool, success: bool) {
        let mut contexts = vec![(c, action)];
        if c.phase == 1 && c.cheat >= self.model.nodes[c.node as usize].steps {
            let effect = if action == 0 { c.observed } else { action };
            contexts = (1..=6)
                .map(|observed| {
                    let equivalent = Context { observed, ..c };
                    (equivalent, if effect == observed { 0 } else { effect })
                })
                .collect();
        }
        for (context, action) in contexts {
            let key = self.model.key(context, win);
            self.model.frozen[win as usize].insert(key, action);
            if win {
                self.hits.remove(&key);
                if success {
                    self.hits.insert(key);
                }
            }
        }
    }

    fn train(&mut self, c: Context, total: u64, win: bool, r: &Request, poll: Poll) -> u64 {
        let actions = self.actions(c);
        if actions.is_empty() {
            return 0;
        }
        let total = total.max(actions.len() as u64 * if win { 2 } else { 1 });
        let base = r.seed as i128 + self.total_control as i128 * 104_729;
        let batches = if win {
            vec![(total.div_ceil(2), false), (total / 2, true)]
        } else {
            vec![(total, false)]
        };
        let mut completed = 0;
        for (sequence, (budget, win)) in batches.into_iter().enumerate() {
            let mut actions = actions.clone();
            if win {
                // Python 的 defaultdict 查询也会建立零计数项，稳定排序保留合法动作顺序。
                actions.sort_by_key(|&a| {
                    self.model.q[1]
                        .entry(self.model.q_key(c, a, true))
                        .or_default()
                        .count
                });
            }
            let mut rng = Random::new(seed(
                base + sequence as i128 * 1_000_003 + if win { 900_000_007 } else { 0 },
            ));
            let mut cache = [ActionCache::new(false), ActionCache::new(false)];
            for index in 0..budget {
                if poll(completed, total, 0) != 0 {
                    self.total_control += completed;
                    return completed;
                }
                let fraction = index as f64 / budget.saturating_sub(1).max(1) as f64;
                let epsilon = r.epsilon_start + (r.epsilon_end - r.epsilon_start) * fraction;
                self.model.rollout(
                    c,
                    actions[index as usize % actions.len()],
                    &mut rng,
                    epsilon,
                    true,
                    win,
                    &mut cache,
                );
                completed += 1;
            }
        }
        self.total_control += completed;
        poll(completed, total, 0);
        completed
    }

    fn evaluate(
        &mut self,
        c: Context,
        budget: u64,
        win: bool,
        target: Option<f64>,
        r: &Request,
        poll: Poll,
    ) -> Batch {
        let actions = self.actions(c);
        let mut out = Batch {
            actions,
            stats: [Stats::default(); 64],
            completed: 0,
            expected: 0,
        };
        if out.actions.is_empty() {
            return out;
        }
        let per_action = (budget / out.actions.len() as u64).max(1);
        out.expected = per_action * out.actions.len() as u64;
        let mut cache = [ActionCache::new(true), ActionCache::new(true)];
        for _ in 0..per_action {
            let common = Random::new(seed(
                r.seed as i128 + 500_000_003 + self.total_trials as i128 * 130_363,
            ));
            self.total_trials += 1;
            for (slot, &action) in out.actions.iter().enumerate() {
                if poll(out.completed, out.expected, 1) != 0 {
                    return out;
                }
                let value =
                    self.model
                        .rollout(c, action, &mut common.clone(), 0.0, false, win, &mut cache);
                out.stats[slot].append(value, target);
                out.completed += 1;
            }
        }
        poll(out.completed, out.expected, 1);
        out
    }

    fn winner(
        &self,
        c: Context,
        batch: &Batch,
        win: bool,
        incumbent: u8,
        supported: bool,
    ) -> usize {
        let mut selected = 0;
        for i in 1..batch.actions.len() {
            let rank = |slot: usize| {
                let s = &batch.stats[slot];
                (
                    if win { rate(s) } else { 0.0 },
                    win && batch.actions[slot] == incumbent && (supported || s.wins > 0),
                    mean(s) + self.model.bonus(c, batch.actions[slot]),
                )
            };
            let a = rank(i);
            let b = rank(selected);
            if a > b || (a == b && batch.actions[i] < batch.actions[selected]) {
                selected = i;
            }
        }
        selected
    }

    fn current(
        &mut self,
        c: Context,
        budget: u64,
        target: Option<f64>,
        r: &Request,
        poll: Poll,
    ) -> Report {
        let mut out = Report::default();
        let count = self.actions(c).len();
        if count == 0 {
            return out;
        }
        let incumbent = if target.is_some() {
            self.choose(c, true).unwrap()
        } else {
            255
        };
        let supported = target.is_some() && self.evidence(c, incumbent);
        let mean_budget = if target.is_some() {
            (budget / 2).max(count as u64)
        } else {
            budget
        };
        let batch = self.evaluate(c, mean_budget, false, target, r, poll);
        out.evaluation = batch.completed;
        out.count = count as u32;
        out.actions[..count].copy_from_slice(&batch.actions);
        out.mean = batch.stats;
        if batch.completed != batch.expected {
            out.cancelled = 1;
            return out;
        }
        let recommended = batch.actions[self.winner(c, &batch, false, 255, false)];
        out.recommended = recommended as u32;
        out.highest_win = recommended as u32;
        if target.is_some() {
            let wins = self.evaluate(
                c,
                budget.saturating_sub(mean_budget).max(count as u64),
                true,
                target,
                r,
                poll,
            );
            out.evaluation += wins.completed;
            out.wins = wins.stats;
            out.has_wins = 1;
            if wins.completed != wins.expected {
                out.cancelled = 1;
                return out;
            }
            let slot = self.winner(c, &wins, true, incumbent, supported);
            let winner = wins.actions[slot];
            out.highest_win = winner as u32;
            self.freeze(c, recommended, false, false);
            self.freeze(
                c,
                winner,
                true,
                wins.stats[slot].wins > 0 || (winner == incumbent && supported),
            );
        } else {
            self.freeze(c, recommended, false, false);
        }
        out
    }

    fn successors(&mut self, c: Context, r: &Request) -> Vec<Context> {
        let base = r.seed as i128 + 700_000_003 + self.total_control as i128 * 130_363;
        let mut successors = Vec::new();
        let mut seen = HashSet::new();
        for action in self.actions(c) {
            for probe in 0..6 {
                let mut next =
                    self.model
                        .advance(c, action, &mut Random::new(seed(base + probe * 104_729)));
                if next.phase == 1 {
                    next.observed = probe as u8 + 1;
                }
                let actions = self.actions(next);
                if actions.len() == 1 {
                    self.model.frozen[0].insert(self.model.key(next, false), actions[0]);
                    self.model.frozen[1].insert(self.model.key(next, true), actions[0]);
                } else if !actions.is_empty() && seen.insert(next) {
                    successors.push(next);
                }
            }
        }
        successors
    }

    fn allocate(&self, successors: &[Context], total: u64) -> Vec<u64> {
        let weights: Vec<u64> = successors
            .iter()
            .map(|&c| self.actions(c).len() as u64)
            .collect();
        let sum: u64 = weights.iter().sum();
        let total = total.max(sum);
        let mut budgets: Vec<u64> = weights.iter().map(|w| total / sum * w).collect();
        for index in 0..total % sum {
            budgets[index as usize % weights.len()] += 1;
        }
        budgets
    }

    fn calibrate(
        &mut self,
        successors: &[Context],
        control: u64,
        evaluation: u64,
        target: Option<f64>,
        r: &Request,
        poll: Poll,
    ) -> (u64, u64, bool) {
        if successors.is_empty() {
            return (0, 0, false);
        }
        let (mut trained, mut evaluated) = (0, 0);
        for (&c, budget) in successors.iter().zip(self.allocate(successors, control)) {
            if poll(trained, control, 2) != 0 {
                return (trained, evaluated, true);
            }
            trained += self.train(c, budget, target.is_some(), r, poll);
        }
        let budgets = self.allocate(successors, evaluation / 2);
        let mut remaining = evaluation.saturating_sub(budgets.iter().sum());
        let mut groups: Vec<(Context, u64)> = Vec::new();
        let mut indices = HashMap::new();
        for visit in &self.model.visits {
            let mut key = self.model.key(visit.context, true);
            if visit.context.phase == 1
                && visit.context.cheat >= self.model.nodes[key.node as usize].steps
            {
                key.observed = 0;
            }
            let index = *indices.entry(key).or_insert_with(|| {
                groups.push((visit.context, 0));
                groups.len() - 1
            });
            groups[index].1 += visit.count;
        }
        // 两次稳定排序保持 Python dict 的首次访问顺序和平局顺序。
        groups.sort_by_key(|item| std::cmp::Reverse(item.1));
        let mut selected = Vec::new();
        for (c, _) in groups {
            let cost = self.actions(c).len() as u64 * 4;
            if cost <= remaining {
                selected.push((c, cost));
                remaining -= cost;
            }
        }
        selected.sort_by_key(|&(c, _)| {
            (
                self.model.nodes[c.node as usize].steps,
                self.model.key(c, false).cheat,
                c.reroll,
                -(c.phase as i32 - 1),
            )
        });
        for (c, budget) in selected {
            if poll(evaluated, evaluation, 2) != 0 {
                return (trained, evaluated, true);
            }
            let incumbent = self.choose(c, target.is_some()).unwrap();
            let supported = target.is_some() && self.evidence(c, incumbent);
            let batch = self.evaluate(c, budget, target.is_some(), target, r, poll);
            evaluated += batch.completed;
            if batch.completed != batch.expected {
                return (trained, evaluated, true);
            }
            let slot = self.winner(c, &batch, true, incumbent, supported);
            let winner = batch.actions[slot];
            self.freeze(
                c,
                winner,
                target.is_some(),
                batch.stats[slot].wins > 0 || (winner == incumbent && supported),
            );
        }
        for (&c, budget) in successors.iter().zip(budgets) {
            if poll(evaluated, evaluation, 2) != 0 {
                return (trained, evaluated, true);
            }
            let report = self.current(c, budget, target, r, poll);
            evaluated += report.evaluation;
            if report.cancelled != 0 {
                return (trained, evaluated, true);
            }
        }
        (trained, evaluated, false)
    }

    fn recommend(&mut self, r: &Request, poll: Poll) -> Report {
        let target = (r.has_target != 0).then_some(r.target);
        self.model.clear_visits();
        self.model.frozen.iter_mut().for_each(|p| p.clear());
        self.hits.clear();
        if self.model.target != target {
            self.model.q[1].clear();
            self.model.target = target;
        }
        let successors = self.successors(r.initial, r);
        let count = self.actions(r.initial).len() as u64;
        let weight: u64 = successors
            .iter()
            .map(|&c| self.actions(c).len() as u64)
            .sum();
        let (control, next_control, evaluation, next_evaluation) = if successors.is_empty() {
            (r.control.max(r.min_visits * count), 0, r.evaluation, 0)
        } else {
            let control = (r.control / 2).max(r.min_visits * count);
            let evaluation = (r.evaluation / 4).max(count);
            (
                control,
                r.control.saturating_sub(control).max(weight),
                evaluation,
                r.evaluation.saturating_sub(evaluation).max(weight),
            )
        };
        let trained = self.train(r.initial, control, target.is_some(), r, poll);
        let (branch_control, branch_evaluation, cancelled) =
            self.calibrate(&successors, next_control, next_evaluation, target, r, poll);
        self.model.clear_visits();
        let mut out = if cancelled || poll(0, evaluation, 2) != 0 {
            Report {
                cancelled: 1,
                ..Report::default()
            }
        } else {
            self.current(r.initial, evaluation, target, r, poll)
        };
        out.control = trained + branch_control;
        out.evaluation += branch_evaluation;
        out
    }

    pub fn execute(&mut self, r: &Request, poll: Poll) -> Report {
        self.model.reliable = r.min_visits.min(2);
        if r.mode != 0 {
            self.model.target = (r.has_learning_target != 0).then_some(r.learning_target);
        }
        let target = (r.has_target != 0).then_some(r.target);
        let mut out = match r.mode {
            0 => self.recommend(r, poll),
            1 => Report {
                control: self.train(r.initial, r.control, r.train_win != 0, r, poll),
                ..Report::default()
            },
            2 => {
                let batch = self.evaluate(r.initial, r.evaluation, r.win != 0, target, r, poll);
                let mut out = Report {
                    mean: batch.stats,
                    count: batch.actions.len() as u32,
                    evaluation: batch.completed,
                    cancelled: (batch.completed != batch.expected) as u32,
                    ..Report::default()
                };
                out.actions[..batch.actions.len()].copy_from_slice(&batch.actions);
                out
            }
            3 => self.current(r.initial, r.evaluation, target, r, poll),
            4 => {
                let mut out = Report::default();
                let mut cache = [ActionCache::new(true), ActionCache::new(true)];
                for index in 0..r.evaluation {
                    if poll(index, r.evaluation, 1) != 0 {
                        out.cancelled = 1;
                        break;
                    }
                    let mut rng = Random::new(seed(
                        r.seed as i128 + 500_000_003 + self.total_trials as i128 * 130_363,
                    ));
                    self.total_trials += 1;
                    let c = Context {
                        observed: (index % 6 + 1) as u8,
                        ..r.initial
                    };
                    let value = self.model.rollout(
                        c,
                        r.first_actions[index as usize % 6],
                        &mut rng,
                        0.0,
                        false,
                        false,
                        &mut cache,
                    );
                    out.mean[0].append(value, target);
                    out.evaluation += 1;
                }
                out
            }
            _ => panic!("unknown evaluation operation"),
        };
        out.total_control = self.total_control;
        out.total_trials = self.total_trials;
        out.q_states = self.model.q[0].len() as u64;
        out
    }
}
