//! Python 仅传入地图/请求、接收根动作统计；完整计算在持久 Rust 会话内完成。
mod evaluator;
mod model;
mod rng;

use evaluator::{Engine, Poll, Report, Request};
use model::{Context, Model, Node, Policy, Record, Visit};
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
    engine: Engine,
    records: Vec<Record>,
    policies: Vec<Policy>,
}

#[no_mangle]
pub extern "C" fn countdown_abi() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn countdown_layout(index: u32) -> usize {
    [
        size_of::<Context>(),
        size_of::<Node>(),
        size_of::<model::Stats>(),
        size_of::<Record>(),
        size_of::<Policy>(),
        size_of::<Visit>(),
        size_of::<Request>(),
        size_of::<Report>(),
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
            engine: Engine::new(Model::new(nodes, reliable)),
            records: Vec::new(),
            policies: Vec::new(),
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_destroy(session: *mut Session) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}

// 只用于首次接管既有 Python 状态；正常推荐不反复交换中间表。
#[no_mangle]
pub unsafe extern "C" fn countdown_load(
    session: *mut Session,
    records: *const Record,
    count: usize,
) -> i32 {
    guarded(|| {
        for record in slice::from_raw_parts(records, count) {
            (*session).engine.model.q[record.mode as usize].insert(record.key, record.stats);
        }
        1
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_initialize(
    session: *mut Session,
    policies: *const Policy,
    count: usize,
    control: u64,
    trials: u64,
    target: f64,
    has_target: u32,
) -> i32 {
    guarded(|| {
        let engine = &mut (*session).engine;
        engine.total_control = control;
        engine.total_trials = trials;
        engine.model.target = (has_target != 0).then_some(target);
        for policy in slice::from_raw_parts(policies, count) {
            engine.model.frozen[(policy.mode & 1) as usize].insert(policy.key, policy.action as u8);
            if policy.mode & 2 != 0 {
                engine.hits.insert(policy.key);
            }
        }
        1
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_execute(
    session: *mut Session,
    request: *const Request,
    output: *mut Report,
    poll: Poll,
) -> i32 {
    guarded(|| {
        assert!(!session.is_null() && !request.is_null() && !output.is_null());
        *output = (*session).engine.execute(&*request, poll);
        1
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_choose(
    session: *mut Session,
    context: *const Context,
    win: u32,
) -> i32 {
    guarded(|| {
        (*session)
            .engine
            .choose(*context, win != 0)
            .map_or(-1, i32::from)
    })
}

// 以下快照仅供测试/显式回退使用；正常调用不访问这些导出函数。
#[no_mangle]
pub unsafe extern "C" fn countdown_records(
    session: *mut Session,
    count: *mut usize,
) -> *const Record {
    guarded(|| {
        let session = &mut *session;
        session.records.clear();
        for (mode, table) in session.engine.model.q.iter().enumerate() {
            session
                .records
                .extend(table.iter().map(|(&key, &stats)| Record {
                    key,
                    stats,
                    mode: mode as u32,
                }));
        }
        *count = session.records.len();
        session.records.as_ptr()
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_snapshot_policies(
    session: *mut Session,
    count: *mut usize,
) -> *const Policy {
    guarded(|| {
        let session = &mut *session;
        session.policies.clear();
        for (mode, table) in session.engine.model.frozen.iter().enumerate() {
            session
                .policies
                .extend(table.iter().map(|(&key, &action)| Policy {
                    key,
                    action: action as u32,
                    mode: mode as u32
                        | if mode == 1 && session.engine.hits.contains(&key) {
                            2
                        } else {
                            0
                        },
                }));
        }
        *count = session.policies.len();
        session.policies.as_ptr()
    })
}

#[no_mangle]
pub unsafe extern "C" fn countdown_visits(
    session: *mut Session,
    count: *mut usize,
) -> *const Visit {
    let visits = &(*session).engine.model.visits;
    *count = visits.len();
    visits.as_ptr()
}
