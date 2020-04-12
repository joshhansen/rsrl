use std::{cell::{RefCell}, rc::Rc};

pub fn make_shared<T>(t: T) -> Shared<T> { Rc::new(RefCell::new(t)) }

pub type Shared<T> = Rc<RefCell<T>>;