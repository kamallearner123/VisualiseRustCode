# Rust Visual Memory Debugger - Usage Guide

## Quick Start

### 1. Access the Application
Open your browser and navigate to `http://localhost:8000`

### 2. Choose Your Execution Mode

The application offers two execution modes:

#### 🟢 Normal Mode (Default: OFF)
- **Purpose**: Quick code execution and testing
- **Speed**: Fast (uses stable Rust compiler)
- **Output**: Shows stdout and stderr only
- **Use When**:
  - Testing basic functionality
  - Checking program output
  - Quick iterations during development
  - No need for memory analysis

#### 🔵 Miri Mode (Default: ON)
- **Purpose**: Memory safety analysis and visualization
- **Speed**: Slower (interprets code with Miri)
- **Output**: Shows stdout, stderr, AND memory visualization
- **Use When**:
  - Debugging memory issues
  - Understanding ownership and borrowing
  - Visualizing stack and heap allocations
  - Learning Rust memory model

### 3. Toggle Between Modes

Use the toggle switch in the editor panel header:
- **Left (Normal)**: Standard execution
- **Right (Miri)**: Memory analysis mode

The run button text will update to reflect the selected mode:
- Normal Mode: "▶ Run"
- Miri Mode: "▶ Run with Miri"

## Features by Mode

### Normal Mode Features
✅ Fast execution  
✅ Stdout/stderr output  
✅ Syntax error detection  
✅ Compilation errors  
❌ Memory visualization  
❌ Ownership tracking  
❌ Pointer analysis  

### Miri Mode Features
✅ Stdout/stderr output  
✅ Syntax error detection  
✅ Compilation errors  
✅ **Memory visualization**  
✅ **Ownership tracking**  
✅ **Pointer analysis**  
✅ **Stack frame visualization**  
✅ **Heap allocation tracking**  
⚠️ Slower execution  

## Example Workflows

### Workflow 1: Quick Testing
```rust
fn main() {
    let result = 2 + 2;
    println!("2 + 2 = {}", result);
}
```
1. Switch to **Normal Mode**
2. Click **Run** (or Ctrl+Enter)
3. View output instantly

### Workflow 2: Memory Analysis
```rust
fn main() {
    let x = Box::new(42);
    let y = &x;
    println!("x = {}, y = {}", x, y);
}
```
1. Switch to **Miri Mode**
2. Click **Run with Miri** (or Ctrl+Enter)
3. View output in Output panel
4. View memory layout in Memory panel
5. See stack variables and heap allocations
6. Observe pointer relationships

### Workflow 3: Debugging Ownership
```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;  // s1 moved to s2
    // println!("{}", s1);  // This would error
    println!("{}", s2);
}
```
1. Use **Miri Mode**
2. Run the code
3. Observe in Memory panel:
   - `s1` marked as **moved** (red)
   - `s2` marked as **owned** (green)
   - Heap allocation shown with pointer from `s2`

## Understanding the Output

### Output Panel

Shows execution results with mode badge:

```
[NORMAL MODE] or [MIRI MODE]

Standard Output:
<program output>

Standard Error:
<error messages>
```

### Memory Panel (Miri Mode Only)

#### Stack Section
- Shows function call frames
- Lists local variables
- Displays addresses and sizes
- Color codes by ownership state

#### Heap Section
- Shows dynamic allocations
- Lists Box, Vec, String, etc.
- Shows allocation addresses
- Displays current values

#### Pointers & References Section
- Shows reference relationships
- Displays borrow types (& or &mut)
- Visual arrows from source to target

### Ownership Colors

| Color | State | Meaning |
|-------|-------|---------|
| 🟢 Green | Owned | Variable owns the value |
| 🔵 Blue | Borrowed | Immutable reference (&T) |
| 🟣 Purple | Borrowed Mut | Mutable reference (&mut T) |
| 🔴 Red | Moved | Value moved, no longer accessible |

## Keyboard Shortcuts

- `Ctrl+Enter` or `Cmd+Enter` - Run code in selected mode
- `Ctrl+K` or `Cmd+K` - Clear editor and output

## Tips and Best Practices

### When to Use Normal Mode
1. **Rapid prototyping** - Quick tests without memory concerns
2. **Output verification** - Just checking print statements
3. **Algorithm testing** - Testing logic without memory analysis
4. **Large computations** - When execution speed matters

### When to Use Miri Mode
1. **Learning Rust** - Understanding ownership and borrowing
2. **Debugging** - Investigating memory-related issues
3. **Teaching** - Demonstrating Rust memory concepts
4. **Code review** - Verifying memory safety patterns

### Performance Considerations

**Normal Mode:**
- Execution time: ~1-2 seconds for simple programs
- Network latency: Minimal
- Resource usage: Low

**Miri Mode:**
- Execution time: ~3-10 seconds for simple programs
- Network latency: Higher (more data transfer)
- Resource usage: Moderate
- Note: Complex programs may timeout

### Troubleshooting

#### "Execution timed out"
- **Problem**: Code is too complex or has infinite loop
- **Solutions**:
  - Simplify the code
  - Use Normal Mode for initial testing
  - Check for infinite loops
  - Reduce iteration counts

#### "No memory visualization"
- **Problem**: Using Normal Mode
- **Solution**: Switch to Miri Mode for memory visualization

#### "Network Error"
- **Problem**: Connection to Rust Playground failed
- **Solutions**:
  - Check internet connection
  - Wait a moment and retry
  - Check if playground is down: https://play.rust-lang.org

## Advanced Usage

### Comparing Modes

You can compare behavior between modes:
1. Run code in Normal Mode, note the output
2. Switch to Miri Mode, run again
3. Compare execution time and behavior
4. Use Memory panel to understand why certain behaviors occur

### Example: Understanding Box
```rust
fn main() {
    let x = 5;           // Stack
    let y = Box::new(5); // Heap
    println!("x = {}, y = {}", x, y);
}
```

**Normal Mode Result:**
- Shows: `x = 5, y = 5`
- Time: <1 second

**Miri Mode Result:**
- Shows: `x = 5, y = 5`
- Time: ~3 seconds
- Memory panel shows:
  - Stack: variable `x` at 0x1000
  - Heap: Box allocation at 0x2000
  - Pointer from `y` (stack) to heap allocation

## FAQ

**Q: Why is Miri mode slower?**  
A: Miri interprets every operation to track memory, while normal mode compiles to native code.

**Q: Can I use both modes for the same code?**  
A: Yes! Toggle the switch anytime to run in different modes.

**Q: Will I always get memory visualization?**  
A: Only in Miri mode. Normal mode is for quick execution only.

**Q: What if my code doesn't compile?**  
A: Both modes show compilation errors in the output panel.

**Q: Can I save my code?**  
A: Currently, code is session-based. Copy/paste to save externally.

## Support

For issues or questions:
1. Check the main README.md
2. Open an issue on GitHub
3. Consult Rust documentation: https://doc.rust-lang.org

---

Happy Rust programming! 🦀
