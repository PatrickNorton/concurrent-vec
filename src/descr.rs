use crate::{FvdVec, LIMIT};
use crossbeam::epoch::{self, Atomic, Guard, Owned, Pointer, Shared};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

const UNDECIDED: u8 = 0;
const FAILED: u8 = 1;
const PASSED: u8 = 2;

/// A holder for the value in a `FvdVec`.
///
/// Note that the variant in use should be determined by the tag on the
/// pointer to this, for safety.
pub union Value<T> {
    data: ManuallyDrop<Node<T>>,
    push: ManuallyDrop<Descriptor<T>>,
}

pub enum ValueEnum<T> {
    Data(Node<T>),
    Push(Descriptor<T>),
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
    // Once `self.state` is inhabited by `FAILED` or `PASSED`, then this
    // descriptor will not access `self.value`. In particular, this means that
    // `drop()` will never be called on the inhabitant of `value`.
    value: Atomic<Value<T>>,
}

pub struct PopDescr {}

pub struct PopSubDescr {}

#[repr(align(2))]
struct Align2;

impl<T> Value<T> {
    pub fn new_data(value: T) -> Value<T> {
        Value {
            data: ManuallyDrop::new(Node::new(value)),
        }
    }

    pub fn new_descriptor(value: Descriptor<T>) -> Value<T> {
        Value {
            push: ManuallyDrop::new(value),
        }
    }

    /// Takes the `Value` and adds it to the list of things to be garbage-
    /// collected.
    ///
    /// SAFETY: It must not be possible to access the guard through a later
    /// epoch (same as [`Guard::defer_drop`]. The `Shared` must also follow the
    /// tag convention. It may, however, be null, in which case nothing will
    /// happen.
    pub unsafe fn defer_drop(this: Shared<Value<T>>, guard: &Guard) {
        if !this.is_null() {
            guard.defer_unchecked(move || drop(Value::into_enum(this.into_owned())))
        }
    }

    /// Takes the `Value` and turns it into an enum.
    ///
    /// SAFETY: The `Owned` must follow the tag convention for
    /// `Owned<Value<T>>`.
    #[allow(clippy::wrong_self_convention)]
    pub unsafe fn into_enum(this: Owned<Value<T>>) -> ValueEnum<T> {
        if this.tag() == 0 {
            ValueEnum::Data(ManuallyDrop::into_inner(this.into_box().data))
        } else {
            ValueEnum::Push(ManuallyDrop::into_inner(this.into_box().push))
        }
    }

    /// Takes the `Value` and turns it into its data component.
    ///
    /// SAFETY: This must be known to contain a `Node`, and may not contain a
    /// descriptor.
    pub unsafe fn into_data(self) -> Node<T> {
        ManuallyDrop::into_inner(self.data)
    }

    pub unsafe fn as_descriptor(&self) -> &Descriptor<T> {
        &self.push
    }
}

impl<T> Node<T> {
    pub fn new(value: T) -> Node<T> {
        Node {
            val: value,
            ref_count: AtomicUsize::new(1),
            _align: Align2,
        }
    }
}

impl<T> Descriptor<T> {
    /// Calls the `complete()` function for the given descriptor.
    ///
    /// SAFETY: To be safe, `this.deref()` must be safe, and `this.deref()`
    /// must point to a valid Descriptor value.
    pub unsafe fn complete_unchecked(
        this: Shared<Value<T>>,
        pos: usize,
        value: &FvdVec<T>,
    ) -> bool {
        assert!(is_descr(this));
        this.deref().push.complete_inner(pos, value, this)
    }

    fn complete_inner(&self, pos: usize, value: &FvdVec<T>, shared_self: Shared<Value<T>>) -> bool {
        match self {
            Descriptor::Push(push) => push.complete(pos, value, shared_self),
            Descriptor::Pop(pop) => pop.complete(pos, value),
            Descriptor::PopSub(pop_sub) => pop_sub.complete(pos, value),
        }
    }
}

impl<T> PushDescr<T> {
    /// SAFETY: The tag convention must be followed and `value` must be
    /// inhabited by a `Node`. Furthermore, if `self.complete()` returns
    /// `true`, then this *must* have exclusive access to `value`. If it
    /// returns `false`, then access is relinquished and another thread
    /// may access it, e.g. in order to call `drop()`.
    pub unsafe fn new_value(value: Atomic<Value<T>>) -> PushDescr<T> {
        PushDescr {
            state: AtomicU8::new(UNDECIDED),
            value,
        }
    }

    pub fn complete(&self, pos: usize, value: &FvdVec<T>, shared_self: Shared<Value<T>>) -> bool {
        assert_eq!(self as *const _ as usize, shared_self.into_usize());
        let guard = &epoch::pin();
        let spot = value.get_spot(pos, guard);
        let spot_2 = value.get_spot(pos - 1, guard);
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
            // SAFETY: We know current is a descriptor as per the `is_descr()`
            // call earlier.
            // This is safe to dereference because if is_descr() succeeds, then
            // the Shared points to a valid descriptor.
            // NOTE: We don't deal with the descriptor at all because it's the
            // job of the thread that placed the descriptor to clean up after
            // it.
            unsafe { Descriptor::complete_unchecked(current, pos - 1, value) };
            current = spot_2.load(Ordering::SeqCst, guard);
        }
        if self.state.load(Ordering::SeqCst) == UNDECIDED {
            let _ = self.state.compare_exchange(
                UNDECIDED,
                if !current.is_null() { PASSED } else { FAILED },
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
        }
        if self.state.load(Ordering::SeqCst) == PASSED {
            // NOTE: While there is no `unsafe` here, it should be noted that
            // once this is passed over, we no longer "own" the value, so we
            // may not do anything with it, including drop it.
            // NOTE 2: While I originally thought this could be a memory leak
            // if `spot` is swapped out  with a different value (e.g. not the
            // value in `self.value`) before this compare_exchange, I think
            // this is not the case, since any thread planning to insert a
            // value and detects a descriptor will call `complete` on it. Note
            // that if this isn't the case, and in intermediate value is
            // swapped in between the store of `self.state` and this
            // compare_exchange, a memory leak will occur.
            if spot
                .compare_exchange(
                    shared_self,
                    self.value.load(Ordering::SeqCst, guard),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                )
                .is_ok()
            {
                // SAFETY: Once we've removed this from the vector, it can no
                // longer be accessed by anybody new. It also follows the tag
                // convention.
                unsafe { Value::defer_drop(shared_self, guard) }
            }
        } else {
            // Makes sure this is no longer accessible, so it can be cleaned
            // up, so the given thread can take care of it.
            if spot
                .compare_exchange(
                    shared_self,
                    Shared::null(),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    guard,
                )
                .is_ok()
            {
                // SAFETY: Once we've removed this from the vector, it can no
                // longer be accessed by anybody new. It also follows the tag
                // convention.
                unsafe { Value::defer_drop(shared_self, guard) }
            }
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
pub fn is_descr<T>(current: Shared<Value<T>>) -> bool {
    current.tag() == 1 && !current.is_null()
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
