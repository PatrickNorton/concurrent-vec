use crate::FvdVec;
use crossbeam::atomic::AtomicCell;
use crossbeam::epoch;
use crossbeam::epoch::{Owned, Pointer, Shared};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

const UNDECIDED: u8 = 0;
const FAILED: u8 = 1;
const PASSED: u8 = 2;

const LIMIT: usize = 32;

const NOT_VALUE: usize = usize::MAX;

/// A holder for the value in a `FvdVec`.
///
/// Note that the variant in use should be determined by the tag on the
/// pointer to this, for safety.
pub union Value<T> {
    data: ManuallyDrop<Node<T>>,
    push: ManuallyDrop<Descriptor<T>>,
}

pub struct Node<T> {
    pub val: T,
    ref_count: AtomicUsize,
    _align: Align2,
}

pub enum Descriptor<T> {
    Push(PushDescr<T>),
    Pop(PopDescr),
    PopSub(PopSubDescr),
}

pub struct PushDescr<T> {
    state: AtomicU8,
    // SAFETY: self.value may *not* contain a descriptor (this is here to
    // prevent reallocation when the value is moved into the vec).
    value: AtomicCell<Option<Box<Value<T>>>>,
}

pub struct PopDescr {}

pub struct PopSubDescr {}

#[repr(align(2))]
struct Align2;

impl<T> Descriptor<T> {
    pub fn complete(&self, pos: usize, value: &FvdVec<T>) -> bool {
        match self {
            Descriptor::Push(push) => push.complete(pos, value, Shared::null()),
            Descriptor::Pop(pop) => pop.complete(pos, value),
            Descriptor::PopSub(pop_sub) => pop_sub.complete(pos, value),
        }
    }
}

impl<T> PushDescr<T> {
    pub fn complete(&self, pos: usize, value: &FvdVec<T>, shared_self: Shared<Value<T>>) -> bool {
        assert_eq!(self as *const _ as usize, shared_self.into_usize());
        let guard = &epoch::pin();
        let spot = value.get_spot(pos);
        let spot_2 = value.get_spot(pos - 1);
        let mut failures = 0;
        let mut current = spot_2.load(Ordering::SeqCst, guard);
        while self.state.load(Ordering::SeqCst) == UNDECIDED && is_descr(current) {
            failures += 1;
            if failures > LIMIT {
                let _ = self.state.compare_exchange(
                    UNDECIDED,
                    FAILED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
            }
            // SAFETY: We know current is a descriptor as per the `is_descr()` call
            // earlier.
            // FIXME? How do we know this is safe to deref in the first place?
            unsafe { current.deref().push.complete(pos - 1, value) };
            current = spot_2.load(Ordering::SeqCst, guard);
        }
        if self.state.load(Ordering::SeqCst) == UNDECIDED {
            let _ = self.state.compare_exchange(
                UNDECIDED,
                if is_value(current) { PASSED } else { FAILED },
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
        }
        if self.state.load(Ordering::SeqCst) == PASSED {
            if let Option::Some(x) = self.value.take() {
                let value = Owned::from(x);
                let _ = spot.compare_exchange(
                    shared_self,
                    value,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                );
            }
        } else {
            let _ = spot.compare_exchange(
                shared_self,
                // SAFETY: Pointers that are NOT_VALUE are never actually used
                unsafe { Shared::from_usize(NOT_VALUE) },
                Ordering::SeqCst,
                Ordering::SeqCst,
                guard,
            );
        }
        self.state.load(Ordering::SeqCst) == PASSED
    }
}

impl PopDescr {
    pub fn complete<T>(&self, pos: usize, value: &FvdVec<T>) -> bool {
        todo!()
    }
}

impl PopSubDescr {
    pub fn complete<T>(&self, pos: usize, value: &FvdVec<T>) -> bool {
        todo!()
    }
}

/// Decomposes a `Shared` into a `Result` of its components.
///
/// In order to be used safely, `value` must be safe to dereference. For
/// safety of that, see [`Shared::deref`][a].
///
/// [a]: Shared::deref
pub unsafe fn decompose<T>(value: Shared<Value<T>>) -> Result<&Node<T>, &Descriptor<T>> {
    if value.tag() == 0 {
        Result::Ok(&value.deref().data)
    } else {
        Result::Err(&value.deref().push)
    }
}

#[inline]
fn is_descr<T>(current: Shared<Value<T>>) -> bool {
    current.tag() == 1 && is_value(current) && !current.is_null()
}

#[inline]
fn is_value<T>(current: Shared<Value<T>>) -> bool {
    current.into_usize() != NOT_VALUE
}

/// Increments the given value atomically if it is not 0.
///
/// This is useful for decrementing reference counts, where if the count is 0,
/// the value is already on the destructor queue.
fn try_increment(num: &AtomicUsize) -> bool {
    num.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
        if x == 0 {
            Option::None
        } else {
            Option::Some(x + 1)
        }
    })
    .is_ok()
}
