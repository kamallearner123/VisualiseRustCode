# Rust Examples Reference Guide

This document provides an overview of all 20 examples available in the Rust Visual Memory Debugger.

## Complete Example List

### 1. Basic Syntax & Variables
**Topic**: Introduction to Rust basics  
**Demonstrates**: Stack allocation, heap allocation (Box, Vec, String), borrowing  
**Key Concepts**: Variables, data types, basic memory model

### 2. Ownership - Move Semantics
**Topic**: Ownership and move semantics  
**Demonstrates**: String move, Copy trait behavior  
**Key Concepts**: Move vs Copy, ownership transfer

### 3. Borrowing - Immutable
**Topic**: Immutable references  
**Demonstrates**: Multiple immutable borrows  
**Key Concepts**: Shared references (&T), read-only access

### 4. Borrowing - Mutable
**Topic**: Mutable references  
**Demonstrates**: Single mutable borrow  
**Key Concepts**: Exclusive references (&mut T), write access

### 5. Box - Heap Allocation
**Topic**: Smart pointer for heap allocation  
**Demonstrates**: Box with primitives, arrays, and strings  
**Key Concepts**: Heap allocation, Box<T>, ownership

### 6. Vectors - Dynamic Arrays
**Topic**: Dynamic arrays on the heap  
**Demonstrates**: Vec operations, indexing, iteration  
**Key Concepts**: Dynamic sizing, push/pop, iteration

### 7. Strings - Owned vs Borrowed
**Topic**: String types in Rust  
**Demonstrates**: String vs &str, concatenation  
**Key Concepts**: String (owned), &str (borrowed), manipulation

### 8. Structs - Custom Types
**Topic**: Creating custom data structures  
**Demonstrates**: Struct definition, instantiation, ownership  
**Key Concepts**: Custom types, field access, struct moves

### 9. Enums - Variants
**Topic**: Enumeration types  
**Demonstrates**: Different enum variants  
**Key Concepts**: Enum definition, variant types, pattern matching

### 10. Option - Handling None
**Topic**: Nullable values in Rust  
**Demonstrates**: Some/None variants, pattern matching  
**Key Concepts**: Option<T>, null safety, matching

### 11. Result - Error Handling
**Topic**: Error handling mechanism  
**Demonstrates**: Ok/Err variants, function returns  
**Key Concepts**: Result<T, E>, error propagation, matching

### 12. Lifetimes - References
**Topic**: Lifetime annotations  
**Demonstrates**: Generic lifetime parameters  
**Key Concepts**: Lifetime 'a, reference validity, function parameters

### 13. Closures - Anonymous Functions
**Topic**: Anonymous functions  
**Demonstrates**: Closure syntax, environment capture, move  
**Key Concepts**: |args| body, capturing, move keyword

### 14. Iterators - Lazy Processing
**Topic**: Lazy evaluation with iterators  
**Demonstrates**: map, collect, sum operations  
**Key Concepts**: Iterator trait, lazy evaluation, transformations

### 15. Rc - Reference Counting
**Topic**: Shared ownership with reference counting  
**Demonstrates**: Rc::new, Rc::clone, strong_count  
**Key Concepts**: Rc<T>, multiple ownership, reference counting

### 16. RefCell - Interior Mutability
**Topic**: Runtime borrow checking  
**Demonstrates**: borrow_mut, interior mutability  
**Key Concepts**: RefCell<T>, runtime checks, mutable access

### 17. Traits - Shared Behavior
**Topic**: Defining shared behavior  
**Demonstrates**: Trait definition, implementation  
**Key Concepts**: Trait, impl, polymorphism

### 18. Pattern Matching
**Topic**: Pattern matching syntax  
**Demonstrates**: Match arms, guards, destructuring  
**Key Concepts**: match, patterns, destructuring

### 19. HashMap - Key-Value Store
**Topic**: Hash table data structure  
**Demonstrates**: insert, get, iteration  
**Key Concepts**: HashMap<K, V>, key-value pairs, lookups

### 20. Threads - Concurrency
**Topic**: Concurrent programming  
**Demonstrates**: thread::spawn, join, sleep  
**Key Concepts**: Threads, concurrency, synchronization

## Usage Tips

### For Beginners
Start with these examples in order:
1. Basic Syntax & Variables
2. Ownership - Move Semantics
3. Borrowing - Immutable
4. Borrowing - Mutable
5. Box - Heap Allocation

### For Intermediate Learners
Focus on these topics:
- Structs
- Enums
- Option & Result
- Lifetimes
- Traits

### For Advanced Topics
Explore these examples:
- Closures
- Iterators
- Rc & RefCell
- Pattern Matching
- Threads

## Memory Visualization

### Best Examples for Memory Visualization (Use Miri Mode)
1. **Ownership - Move Semantics**: See ownership transfer clearly
2. **Box - Heap Allocation**: Visualize stack vs heap
3. **Vectors**: Dynamic array growth on heap
4. **Rc - Reference Counting**: Multiple pointers to same data
5. **Strings**: String vs &str memory layout

### Best Examples for Quick Output (Use Normal Mode)
1. **Pattern Matching**: Logic flow demonstration
2. **Iterators**: Functional transformations
3. **Result - Error Handling**: Error flow
4. **Closures**: Function behavior
5. **Threads**: Concurrent output

## Example Categories

### Memory Management
- Ownership - Move Semantics
- Borrowing (Immutable & Mutable)
- Box - Heap Allocation
- Rc - Reference Counting
- RefCell - Interior Mutability

### Data Structures
- Vectors - Dynamic Arrays
- Strings - Owned vs Borrowed
- Structs - Custom Types
- Enums - Variants
- HashMap - Key-Value Store

### Control Flow & Logic
- Pattern Matching
- Option - Handling None
- Result - Error Handling

### Functional Programming
- Closures - Anonymous Functions
- Iterators - Lazy Processing
- Traits - Shared Behavior

### Advanced Topics
- Lifetimes - References
- Threads - Concurrency

## Quick Reference

| Example # | Topic | Difficulty | Memory Concepts |
|-----------|-------|------------|-----------------|
| 1 | Basic Syntax | Beginner | Stack, Heap basics |
| 2 | Ownership Move | Beginner | Move semantics |
| 3 | Immutable Borrow | Beginner | Shared references |
| 4 | Mutable Borrow | Beginner | Exclusive references |
| 5 | Box | Beginner | Heap allocation |
| 6 | Vectors | Beginner | Dynamic arrays |
| 7 | Strings | Intermediate | String types |
| 8 | Structs | Intermediate | Custom types |
| 9 | Enums | Intermediate | Enum variants |
| 10 | Option | Intermediate | Null safety |
| 11 | Result | Intermediate | Error handling |
| 12 | Lifetimes | Advanced | Reference validity |
| 13 | Closures | Intermediate | Environment capture |
| 14 | Iterators | Intermediate | Lazy evaluation |
| 15 | Rc | Advanced | Shared ownership |
| 16 | RefCell | Advanced | Interior mutability |
| 17 | Traits | Intermediate | Polymorphism |
| 18 | Pattern Match | Intermediate | Control flow |
| 19 | HashMap | Intermediate | Collections |
| 20 | Threads | Advanced | Concurrency |

## Learning Paths

### Path 1: Memory Safety (8 examples)
1. Basic Syntax & Variables
2. Ownership - Move Semantics
3. Borrowing - Immutable
4. Borrowing - Mutable
5. Box - Heap Allocation
6. Lifetimes - References
7. Rc - Reference Counting
8. RefCell - Interior Mutability

### Path 2: Data Structures (6 examples)
1. Basic Syntax & Variables
2. Vectors - Dynamic Arrays
3. Strings - Owned vs Borrowed
4. Structs - Custom Types
5. Enums - Variants
6. HashMap - Key-Value Store

### Path 3: Functional Programming (5 examples)
1. Closures - Anonymous Functions
2. Iterators - Lazy Processing
3. Traits - Shared Behavior
4. Option - Handling None
5. Result - Error Handling

### Path 4: Advanced Rust (5 examples)
1. Lifetimes - References
2. Rc - Reference Counting
3. RefCell - Interior Mutability
4. Pattern Matching
5. Threads - Concurrency

## Example Combinations

Try running these examples in sequence to see related concepts:

**Ownership Series**:
1. Ownership - Move Semantics
2. Borrowing - Immutable
3. Borrowing - Mutable

**Heap Allocation Series**:
1. Box - Heap Allocation
2. Vectors - Dynamic Arrays
3. Strings - Owned vs Borrowed

**Error Handling Series**:
1. Option - Handling None
2. Result - Error Handling
3. Pattern Matching

**Smart Pointers Series**:
1. Box - Heap Allocation
2. Rc - Reference Counting
3. RefCell - Interior Mutability

## Tips for Each Example

### When to Use Normal Mode ⚡
- Pattern Matching
- Closures
- Iterators
- Threads
- Result/Option (logic flow)

### When to Use Miri Mode 🔍
- Ownership examples
- Borrowing examples
- Box, Rc, RefCell
- Vectors and Strings
- Structs (to see memory layout)

---

**Total Examples**: 20  
**Beginner Friendly**: 6  
**Intermediate Level**: 9  
**Advanced Topics**: 5  

Each example is self-contained and can run independently. All examples include comments explaining the key concepts being demonstrated.
