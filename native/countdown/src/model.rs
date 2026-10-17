use crate::rng::Random;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct Context {
    pub mask: u64,
    pub cd: i64,
    pub node: u32,
    pub cheat: u32,
    pub reroll: u32,
    pub phase: u8,
    pub observed: u8,
    pub locked: u8,
    pub action: u8,
}
pub type Key = Context;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Node {
    pub future: u64,
    pub neighbors: u64,
    pub bonus: f64,
    pub steps: u32,
    pub edge_count: u32,
    pub edges: [u8; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub count: u64,
    pub total: f64,
    pub total_sq: f64,
    pub wins: u64,
    pub target_count: u64,
    pub minimum: f64,
    pub maximum: f64,
}
impl Default for Stats {
    fn default() -> Self {
        Self {
            count: 0,
            total: 0.0,
            total_sq: 0.0,
            wins: 0,
            target_count: 0,
            minimum: f64::INFINITY,
            maximum: f64::NEG_INFINITY,
        }
    }
}
impl Stats {
    pub fn append(&mut self, value: f64, target: Option<f64>) {
        self.count += 1;
        self.total += value;
        self.total_sq += value * value;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        if let Some(target) = target {
            self.target_count += 1;
            self.wins += (value >= target) as u64;
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Record {
    pub key: Key,
    pub stats: Stats,
    pub mode: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Policy {
    pub key: Key,
    pub action: u32,
    pub mode: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Visit {
    pub context: Context,
    pub count: u64,
}

pub struct ActionCache {
    enabled: bool,
    entries: HashMap<Key, u8>,
}
impl ActionCache {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            entries: HashMap::new(),
        }
    }
    fn get(&self, key: &Key) -> Option<&u8> {
        self.entries.get(key)
    }
    fn insert(&mut self, key: Key, action: u8) {
        if self.enabled {
            if self.entries.len() >= 250_000 {
                self.entries.clear();
            }
            self.entries.insert(key, action);
        }
    }
}

pub struct Model {
    pub nodes: Vec<Node>,
    pub target: Option<f64>,
    pub reliable: u64,
    pub q: [HashMap<Key, Stats>; 2],
    pub frozen: [HashMap<Key, u8>; 2],
    effects: RefCell<HashMap<(u32, u64), [f64; 6]>>,
    pub dirty: HashSet<(usize, Key)>,
    pub visits: Vec<Visit>,
    visit_indices: HashMap<Key, usize>,
}

fn select_bit(mut bits: u64, mut rank: u64) -> u64 {
    while rank > 0 {
        bits &= bits - 1;
        rank -= 1;
    }
    bits & bits.wrapping_neg()
}

impl Model {
    pub fn new(nodes: Vec<Node>, reliable: u64) -> Self {
        Self {
            nodes,
            target: None,
            reliable,
            q: Default::default(),
            frozen: Default::default(),
            effects: Default::default(),
            dirty: Default::default(),
            visits: Vec::new(),
            visit_indices: Default::default(),
        }
    }

    pub fn key(&self, mut c: Context, win: bool) -> Key {
        let node = &self.nodes[c.node as usize];
        c.mask &= node.future;
        c.cheat = c.cheat.min(
            node.steps
                .saturating_sub((c.phase == 2 || c.phase == 3) as u32),
        );
        if !win {
            c.cd = 0;
        }
        if c.phase != 1 {
            c.observed = 0;
        }
        c.locked = (c.phase == 3 && c.locked == 3) as u8;
        c.action = 0;
        c
    }

    pub fn q_key(&self, mut c: Context, action: u8, win: bool) -> Key {
        if c.phase == 1 && action != 7 {
            let effect = if action == 0 { c.observed } else { action };
            if action != 0 {
                c.cheat -= 1;
            }
            c.phase = 3;
            c.locked = 0;
            let mut key = self.key(c, win);
            key.phase = 4;
            key.observed = effect;
            key
        } else {
            let mut key = self.key(c, win);
            key.action = action;
            key
        }
    }
    pub fn actions(&self, c: Context, actions: &mut [u8; 64]) -> usize {
        match c.phase {
            1 => {
                actions[0] = 0;
                let mut len = 1;
                if c.cheat > 0 {
                    for effect in 1..=6 {
                        if effect != c.observed {
                            actions[len] = effect;
                            len += 1;
                        }
                    }
                }
                if c.reroll > 0 {
                    actions[len] = 7;
                    len += 1;
                }
                len
            }
            2 => {
                let mut bits = self.nodes[c.node as usize].future & !c.mask;
                let mut len = 0;
                while bits != 0 {
                    actions[len] = bits.trailing_zeros() as u8;
                    len += 1;
                    bits &= bits - 1;
                }
                len
            }
            3 => {
                let edges = &self.nodes[c.node as usize].edges();
                actions[..edges.len()].copy_from_slice(edges);
                edges.len()
            }
            _ => 0,
        }
    }

    fn moved(&self, mut c: Context, action: u8) -> Context {
        let next = action as usize;
        if c.locked == 3 {
            c.mask |= self.nodes[next].neighbors & self.nodes[c.node as usize].future;
        }
        c.cd += if c.mask & (1u64 << next) != 0 { 1 } else { -3 };
        c.mask &= self.nodes[next].future;
        c.node = next as u32;
        c
    }

    pub fn advance(&self, mut c: Context, action: u8, rng: &mut Random) -> Context {
        match c.phase {
            1 => {
                if action == 7 {
                    c.reroll -= 1;
                    c.observed = 1 + rng.below(6) as u8;
                    c.locked = 0;
                    return c;
                }
                let effect = if action == 0 {
                    c.observed
                } else {
                    c.cheat -= 1;
                    action
                };
                let future = self.nodes[c.node as usize].future;
                c.observed = 0;
                c.locked = effect;
                c.phase = 3;
                match effect {
                    1 => {
                        let base = c.mask;
                        let mut sources = base & future;
                        while sources != 0 {
                            let source = sources.trailing_zeros() as usize;
                            let candidates = self.nodes[source].neighbors & future & !base;
                            if candidates != 0 {
                                c.mask |= select_bit(
                                    candidates,
                                    rng.below(candidates.count_ones() as u64),
                                );
                            }
                            sources &= sources - 1;
                        }
                    }
                    2 => c.cd += (c.mask & future).count_ones() as i64,
                    4 => {
                        if future & !c.mask != 0 {
                            c.phase = 2;
                        }
                    }
                    5 => {
                        let candidates = future & !c.mask;
                        if candidates != 0 {
                            c.mask |=
                                select_bit(candidates, rng.below(candidates.count_ones() as u64));
                        }
                    }
                    _ => (),
                }
                c
            }
            2 => {
                c.mask |= 1u64 << action;
                c.phase = 3;
                c.observed = 0;
                c.locked = 4;
                c
            }
            3 => {
                c = self.moved(c, action);
                c.locked = 0;
                if self.nodes[c.node as usize].edges().is_empty() {
                    c.phase = 0;
                    c.observed = 0;
                } else {
                    c.phase = 1;
                    c.observed = 1 + rng.below(6) as u8;
                }
                c
            }
            _ => c,
        }
    }

    fn effect_values(&self, c: Context) -> [f64; 6] {
        let node = &self.nodes[c.node as usize];
        let remaining = node.steps.max(1) as f64;
        let active = (c.mask & node.future).count_ones() as f64;
        let mut probabilities = [0.0; 64];
        let mut sources = c.mask & node.future;
        // 保持首次插入顺序，避免浮点求和顺序影响极小的平局差值。
        let mut order = [0usize; 64];
        let mut visited = 0u64;
        let mut count = 0;
        while sources != 0 {
            let source = sources.trailing_zeros() as usize;
            let mut candidates = self.nodes[source].neighbors & node.future & !c.mask;
            if candidates != 0 {
                let probability = 1.0 / candidates.count_ones() as f64;
                while candidates != 0 {
                    let dst = candidates.trailing_zeros() as usize;
                    if visited & (1u64 << dst) == 0 {
                        visited |= 1u64 << dst;
                        order[count] = dst;
                        count += 1;
                    }
                    probabilities[dst] = 1.0 - (1.0 - probabilities[dst]) * (1.0 - probability);
                    candidates &= candidates - 1;
                }
            }
            sources &= sources - 1;
        }
        // 与 Python 浮点 sum 的补偿累加保持一致，避免启发式平局被舍入差异打破。
        let mut total = 0.0_f64;
        let mut compensation = 0.0_f64;
        for &idx in &order[..count] {
            let value = probabilities[idx];
            let next = total + value;
            compensation += if total.abs() >= value.abs() {
                (total - next) + value
            } else {
                (value - next) + total
            };
            total = next;
        }
        let spread = total + compensation;
        let adjacent = node
            .edges()
            .iter()
            .map(|&idx| {
                let next = &self.nodes[idx as usize];
                (next.neighbors & next.future & !c.mask).count_ones()
            })
            .max()
            .unwrap_or(0) as f64;
        let has_target = (node.future & !c.mask != 0) as u8 as f64;
        [
            (active + spread) * (remaining - 1.0),
            active * remaining,
            (active + adjacent) * (remaining - 1.0),
            (active + has_target) * (remaining - 1.0),
            (active + has_target) * (remaining - 1.0),
            active * (remaining - 1.0),
        ]
    }

    fn heuristic(&self, c: Context, action: u8, effects: &[f64; 6]) -> f64 {
        match c.phase {
            1 => {
                let value = if action == 7 {
                    effects.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 0.8
                } else {
                    effects[(if action == 0 { c.observed } else { action } - 1) as usize]
                };
                value - if action > 0 && action < 7 { 0.05 } else { 0.0 }
            }
            2 => {
                (if self.nodes[c.node as usize].edges().contains(&action) {
                    4.0
                } else {
                    0.0
                }) + 0.05 * self.nodes[action as usize].steps as f64
            }
            3 => {
                (if c.mask & (1u64 << action) != 0 {
                    1.0
                } else {
                    -3.0
                }) + 0.05 * self.nodes[action as usize].steps as f64
            }
            _ => 0.0,
        }
    }

    fn bonus(&self, c: Context, action: u8) -> f64 {
        if c.phase == 3 {
            self.nodes[action as usize].bonus
        } else {
            0.0
        }
    }

    fn mean_action(&self, c: Context, actions: &[u8], cache: &mut [ActionCache; 2]) -> u8 {
        let key = self.key(c, false);
        if let Some(&action) = cache[0].get(&key) {
            return action;
        }
        if let Some(&action) = self.frozen[0].get(&key) {
            if actions.contains(&action) {
                cache[0].insert(key, action);
                return action;
            }
        }
        let effects = if c.phase == 1 {
            let effect_key = (c.node, c.mask & self.nodes[c.node as usize].future);
            let cached = self.effects.borrow().get(&effect_key).copied();
            cached.unwrap_or_else(|| {
                let values = self.effect_values(c);
                let mut cache = self.effects.borrow_mut();
                if cache.len() >= 8192 {
                    cache.clear();
                }
                cache.insert(effect_key, values);
                values
            })
        } else {
            [0.0; 6]
        };
        let mut scores = [0.0; 64];
        let mut heuristics = [0.0; 64];
        let mut best = f64::NEG_INFINITY;
        for (i, &action) in actions.iter().enumerate() {
            heuristics[i] = self.heuristic(c, action, &effects);
            scores[i] = self.q[0]
                .get(&self.q_key(c, action, false))
                .filter(|s| s.count >= self.reliable)
                .map_or(heuristics[i], |s| s.total / s.count as f64)
                + self.bonus(c, action);
            best = best.max(scores[i]);
        }
        let mut selected = 255;
        let mut best_heuristic = f64::NEG_INFINITY;
        for (i, &action) in actions.iter().enumerate() {
            if (scores[i] - best).abs() <= 1e-12
                && (heuristics[i] > best_heuristic
                    || (heuristics[i] == best_heuristic && action < selected))
            {
                selected = action;
                best_heuristic = heuristics[i];
            }
        }
        assert_ne!(selected, 255);
        cache[0].insert(key, selected);
        selected
    }

    fn choose(&self, c: Context, actions: &[u8], win: bool, cache: &mut [ActionCache; 2]) -> u8 {
        if !win {
            return self.mean_action(c, actions, cache);
        }
        let key = self.key(c, true);
        if let Some(&action) = cache[1].get(&key) {
            return action;
        }
        let mut selected = None;
        let mut best = f64::NEG_INFINITY;
        if c.phase == 3 && self.target.is_some() {
            for &action in actions {
                if !self.nodes[action as usize].edges().is_empty() {
                    continue;
                }
                let moved = self.moved(c, action);
                if moved.cd as f64 >= self.target.unwrap() {
                    let score = moved.cd as f64 + self.bonus(c, action);
                    if score > best || (score == best && Some(action) < selected) {
                        best = score;
                        selected = Some(action);
                    }
                }
            }
        }
        if selected.is_none() {
            selected = self.frozen[1]
                .get(&key)
                .copied()
                .filter(|a| actions.contains(a));
        }
        if selected.is_none() {
            let mut best_rate = -1.0;
            let mut best_mean = f64::NEG_INFINITY;
            for &action in actions {
                if let Some(stats) = self.q[1].get(&self.q_key(c, action, true)) {
                    if stats.target_count == 0 || stats.wins == 0 {
                        continue;
                    }
                    let rate = stats.wins as f64 / stats.target_count as f64;
                    let mean = stats.total / stats.count as f64 + self.bonus(c, action);
                    if rate > best_rate
                        || (rate == best_rate
                            && (mean > best_mean || (mean == best_mean && Some(action) < selected)))
                    {
                        best_rate = rate;
                        best_mean = mean;
                        selected = Some(action);
                    }
                }
            }
        }
        let action = selected.unwrap_or_else(|| self.mean_action(c, actions, cache));
        cache[1].insert(key, action);
        action
    }

    pub fn clear_visits(&mut self) {
        self.visits.clear();
        self.visit_indices.clear();
        self.dirty.clear();
    }

    pub fn rollout(
        &mut self,
        initial: Context,
        forced: u8,
        rng: &mut Random,
        epsilon: f64,
        learn: bool,
        win: bool,
        cache: &mut [ActionCache; 2],
    ) -> f64 {
        let mut context = initial;
        let mut actions = [0; 64];
        let mut trace = Vec::new();
        let limit = self.nodes.len() * 4 + initial.reroll as usize + 32;
        for index in 0..limit {
            if context.phase == 0 {
                break;
            }
            let count = self.actions(context, &mut actions);
            if count == 0 {
                break;
            }
            if learn {
                let key = self.key(context, true);
                let slot = *self.visit_indices.entry(key).or_insert_with(|| {
                    self.visits.push(Visit { context, count: 0 });
                    self.visits.len() - 1
                });
                self.visits[slot].count += 1;
            }
            let action = if index == 0 {
                forced
            } else if epsilon != 0.0 && rng.random() < epsilon {
                actions[rng.below(count as u64) as usize]
            } else {
                self.choose(context, &actions[..count], win, cache)
            };
            if learn {
                trace.push((self.q_key(context, action, win), context.cd));
            }
            context = self.advance(context, action, rng);
            assert!(index + 1 < limit, "rollout exceeded decision limit");
        }
        let value = context.cd as f64;
        for (key, cd) in trace {
            self.q[win as usize].entry(key).or_default().append(
                if win { value } else { value - cd as f64 },
                if win { self.target } else { None },
            );
            self.dirty.insert((win as usize, key));
        }
        value
    }
}

impl Node {
    fn edges(&self) -> &[u8] {
        &self.edges[..self.edge_count as usize]
    }
}
