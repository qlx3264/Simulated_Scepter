//! 仅供 tool/countdown_native.py 使用的私有 C ABI。
//! 桥接层校验并持有数组和句柄，保证同步调用期间有效且不能并发或重入。
mod model;
mod rng;

use model::{ActionCache, Context, Model, Node, Policy, Record, Stats, Visit};
use rng::Random;
use std::{
    cell::RefCell,
    ffi::{c_char, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    slice,
};

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }

fn guarded<T: Default>(call: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(value) => value,
        Err(error) => {
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .unwrap_or("native countdown panic");
            ERROR.with(|slot| {
                *slot.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap()
            });
            T::default()
        }
    }
}

pub struct Session {
    model: Model,
    records: Vec<Record>,
}

#[repr(C)]
pub struct Job {
    initial: Context,
    seed: i64,
    budget: u64,
    epsilon_start: f64,
    epsilon_end: f64,
    target: f64,
    actions: [u8; 64],
    action_count: u32,
    kind: u32,
    has_target: u32,
}

#[repr(C)]
pub struct ResultStats {
    stats: [Stats; 64],
    completed: u64,
    trials: u64,
}

type Poll = extern "C" fn(u64) -> i32;

#[no_mangle]
pub extern "C" fn countdown_abi() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn countdown_layout(index: u32) -> usize {
    [
        size_of::<Context>(),
        size_of::<Node>(),
        size_of::<Stats>(),
        size_of::<Record>(),
        size_of::<Policy>(),
        size_of::<Visit>(),
        size_of::<Job>(),
        size_of::<ResultStats>(),
    ]
    .get(index as usize)
    .copied()
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn countdown_error() -> *const c_char {
    ERROR.with(|slot| slot.borrow().as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn countdown_create(
    nodes: *const Node,
    count: usize,
    reliable: u64,
) -> *mut Session {
    guarded(|| {
        assert!(!nodes.is_null() && (1..=64).contains(&count));
        let nodes = slice::from_raw_parts(nodes, count).to_vec();
        for node in &nodes {
            assert!(node.edge_count <= 64);
            assert!(node.edges[..node.edge_count as usize]
                .iter()
                .all(|&edge| (edge as usize) < count));
        }
        Box::into_raw(Box::new(Session {
            model: Model::new(nodes, reliable),
            records: Vec::new(),
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_destroy(session: *mut Session) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}

#[no_mangle]
pub unsafe extern "C" fn countdown_load(
    session: *mut Session,
    records: *const Record,
    count: usize,
) -> i32 {
    guarded(|| {
        let model = &mut (*session).model;
        for record in slice::from_raw_parts(records, count) {
            model.q[record.mode as usize].insert(record.key, record.stats);
        }
        1
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_policies(
    session: *mut Session,
    policies: *const Policy,
    count: usize,
    target: f64,
    has_target: u32,
) -> i32 {
    guarded(|| {
        let model = &mut (*session).model;
        model.target = (has_target != 0).then_some(target);
        for policy in &mut model.frozen {
            policy.clear();
        }
        for policy in slice::from_raw_parts(policies, count) {
            model.frozen[policy.mode as usize].insert(policy.key, policy.action as u8);
        }
        1
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_run(
    session: *mut Session,
    job: *const Job,
    output: *mut ResultStats,
    poll: Poll,
) -> i32 {
    guarded(|| {
        let job = &*job;
        let output = &mut *output;
        *output = ResultStats {
            stats: [Stats::default(); 64],
            completed: 0,
            trials: 0,
        };
        let model = &mut (*session).model;
        let learn = job.kind < 2;
        let win = job.kind == 1 || job.kind == 3;
        assert!(job.kind <= 4 && (1..=64).contains(&job.action_count));
        assert!((job.initial.node as usize) < model.nodes.len());
        if learn {
            model.clear_visits();
        }
        let mut cache = [ActionCache::new(!learn), ActionCache::new(!learn)];
        let target = (job.has_target != 0).then_some(job.target);
        let mut rng = Random::new(job.seed);
        for index in 0..job.budget {
            if learn {
                if poll(output.completed) != 0 {
                    return 1;
                }
                let fraction = index as f64 / job.budget.saturating_sub(1).max(1) as f64;
                let epsilon = job.epsilon_start + (job.epsilon_end - job.epsilon_start) * fraction;
                model.rollout(
                    job.initial,
                    job.actions[index as usize % job.action_count as usize],
                    &mut rng,
                    epsilon,
                    true,
                    win,
                    &mut cache,
                );
                output.completed += 1;
            } else {
                let seed = job
                    .seed
                    .checked_add(i64::try_from(index).unwrap().checked_mul(130_363).unwrap())
                    .unwrap();
                let common = Random::new(seed);
                output.trials += 1;
                let action_count = if job.kind == 4 {
                    1
                } else {
                    job.action_count as usize
                };
                for slot in 0..action_count {
                    if poll(output.completed) != 0 {
                        return 1;
                    }
                    rng = common.clone();
                    let mut initial = job.initial;
                    let action = if job.kind == 4 {
                        initial.observed = (index % 6 + 1) as u8;
                        job.actions[index as usize % 6]
                    } else {
                        job.actions[slot]
                    };
                    let value =
                        model.rollout(initial, action, &mut rng, 0.0, false, win, &mut cache);
                    output.stats[slot].append(value, target);
                    output.completed += 1;
                }
            }
        }
        1
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_records(
    session: *mut Session,
    count: *mut usize,
) -> *const Record {
    guarded(|| {
        let session = &mut *session;
        session.records = session
            .model
            .dirty
            .iter()
            .map(|&(mode, key)| Record {
                key,
                stats: session.model.q[mode][&key],
                mode: mode as u32,
            })
            .collect();
        *count = session.records.len();
        session.records.as_ptr()
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_visits(
    session: *mut Session,
    count: *mut usize,
) -> *const Visit {
    let visits = &(*session).model.visits;
    *count = visits.len();
    visits.as_ptr()
}
