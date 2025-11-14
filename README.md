# Rust Visual Memory Debugger

A full-featured Django web application that acts as an **online Rust visual memory debugger**. This tool allows you to write Rust code, execute it with Miri (Rust's interpreter), and visualize memory operations in real-time.

![Rust Visual Memory Debugger](https://img.shields.io/badge/Django-4.2-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Rust](https://img.shields.io/badge/Rust-1.70+-orange)

## 🚀 Features

### Core Functionality
- **📝 Online Rust Code Editor** - Monaco Editor with Rust syntax highlighting
- **⚡ Execute Rust Code Online** - Uses the Rust Playground API
- **🎚️ Dual Execution Modes:**
  - **Normal Mode** - Fast execution for quick testing and output
  - **Miri Mode** - Memory analysis with detailed trace visualization
- **🔍 Miri Integration** - Run code with Miri to capture memory traces
- **📊 Real-time Memory Visualization** - Visual representation of stack and heap
- **🎨 Color-coded Ownership** - Visual indicators for ownership states:
  - 🟢 Green = Owned
  - 🔵 Blue = Borrowed (immutable)
  - 🟣 Purple = Borrowed (mutable)
  - 🔴 Red = Moved
- **🔗 Pointer Visualization** - Arrows showing references and borrowing relationships
- **📱 Responsive UI** - Split-pane layout that works on all screen sizes

### User Interface
- **Three-panel layout:**
  - **Editor Panel** - Write and edit Rust code
  - **Output Panel** - View stdout/stderr
  - **Memory Panel** - Visual memory representation
- **20+ Pre-loaded examples** - Quick access to common Rust patterns:
  - Ownership & Borrowing
  - Smart Pointers (Box, Rc, RefCell)
  - Collections (Vec, String, HashMap)
  - Error Handling (Option, Result)
  - Advanced Topics (Lifetimes, Traits, Closures, Threads)
- **Keyboard shortcuts** - Ctrl/Cmd+Enter to run, Ctrl/Cmd+K to clear

## 📋 Requirements

- Python 3.8 or higher
- Django 4.2
- Modern web browser with JavaScript enabled

## 🛠️ Installation

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/kamallearner123/VisualiseRustCode.git
cd VisualiseRustCode

# Run the setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create a virtual environment
- Install all dependencies
- Run database migrations
- Set up the project

### Manual Setup

If you prefer manual installation:

```bash
# 1. Clone the repository
git clone https://github.com/kamallearner123/VisualiseRustCode.git
cd VisualiseRustCode

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run migrations
python manage.py migrate

# 5. Create superuser (Optional)
python manage.py createsuperuser
```

## 🚀 Running the Application

### Start the Server

```bash
# Quick start (recommended)
./start.sh
```

Or manually:

```bash
source venv/bin/activate
python manage.py runserver
```

### Stop the Server

```bash
# Quick stop
./stop.sh
```

Or press `CTRL+C` in the terminal where the server is running.

### Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

## 🎯 Usage

### Running Code

1. **Write or select code:**
   - Type your Rust code in the editor, or
   - Click "Load Example" to choose from pre-built examples

2. **Choose execution mode:**
   - Toggle between **Normal** and **Miri** mode
   - **Normal Mode**: Fast execution, shows only stdout/stderr
   - **Miri Mode**: Slower but provides memory visualization

3. **Execute:**
   - Click "Run" or "Run with Miri" button (based on selected mode), or
   - Press `Ctrl/Cmd + Enter`

4. **View results:**
   - **Output Panel** - See program output and errors
   - **Memory Panel** - (Miri mode only) Visualize stack frames, heap allocations, and pointers

### Example Code

```rust
fn main() {
    // Stack allocation
    let x = 42;
    
    // Heap allocation with Box
    let boxed = Box::new(100);
    
    // Vector (heap allocation)
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
    
    // Borrowing
    let r = &x;
    
    println!("x = {}", x);
    println!("boxed = {}", boxed);
}
```

### Keyboard Shortcuts

- `Ctrl/Cmd + Enter` - Run code
- `Ctrl/Cmd + K` - Clear editor and output

## 📁 Project Structure

```
VisualiseRustCode/
├── manage.py                          # Django management script
├── requirements.txt                   # Python dependencies
├── rust_debugger_project/            # Main project settings
│   ├── __init__.py
│   ├── settings.py                   # Django settings
│   ├── urls.py                       # Root URL configuration
│   ├── wsgi.py                       # WSGI configuration
│   └── asgi.py                       # ASGI configuration
├── debugger/                         # Main application
│   ├── __init__.py
│   ├── models.py                     # Database models
│   ├── views.py                      # View functions
│   ├── urls.py                       # App URL patterns
│   ├── admin.py                      # Admin configuration
│   ├── apps.py                       # App configuration
│   ├── services/                     # Business logic
│   │   ├── __init__.py
│   │   ├── rust_executor.py         # Rust Playground API client
│   │   └── miri_parser.py           # Miri JSON parser
│   └── templates/
│       └── debugger/
│           └── index.html            # Main UI template
├── static/                           # Static files
│   ├── css/
│   │   └── style.css                # Application styles
│   └── js/
│       ├── editor.js                # Monaco Editor setup
│       ├── memory-visualizer.js     # Memory visualization
│       └── main.js                  # Main application logic
└── db.sqlite3                        # SQLite database
```

## 🔧 Configuration

### Settings

The main configuration is in `rust_debugger_project/settings.py`:

- **DEBUG**: Set to `False` in production
- **ALLOWED_HOSTS**: Add your domain in production
- **SECRET_KEY**: Change this in production

### Database

By default, the application uses SQLite. To use PostgreSQL or MySQL:

1. Update `DATABASES` in `settings.py`
2. Install the appropriate database driver
3. Run migrations

## 🌐 API Endpoints

### Execute Code
```
POST /execute/
Content-Type: application/json

{
  "code": "fn main() { println!(\"Hello\"); }",
  "use_miri": true  // Optional, defaults to true
}
```

**Response (with Miri):**
```json
{
  "execution_id": 1,
  "success": true,
  "stdout": "Hello\n",
  "stderr": "",
  "memory_trace": {
    "stack": [...],
    "heap": [...],
    "pointers": [...],
    "ownership": {...}
  },
  "mode": "miri"
}
```

**Response (normal execution):**
```json
{
  "execution_id": 1,
  "success": true,
  "stdout": "Hello\n",
  "stderr": "",
  "mode": "normal"
}
```

### Get Execution
```
GET /execution/<id>/
```

## 🎨 Memory Visualization

The memory visualizer displays:

### Stack
- Function call frames
- Local variables
- Variable addresses and sizes

### Heap
- Dynamic allocations (`Box`, `Vec`, `String`, etc.)
- Allocation addresses and sizes
- Current values

### Pointers
- Reference relationships
- Borrow types (immutable/mutable)
- Visual arrows connecting references

### Ownership States
- **Owned** (Green) - Value is owned by the variable
- **Borrowed** (Blue) - Immutably borrowed
- **Borrowed Mut** (Purple) - Mutably borrowed
- **Moved** (Red) - Value has been moved

## 🧪 Testing

Run tests with:
```bash
python manage.py test
```

## 📝 Notes

### Miri Limitations

The Rust Playground API has limited Miri support with trace flags. For full Miri functionality:

1. **Local Setup:**
   ```bash
   cargo +nightly miri run
   MIRIFLAGS="-Zmiri-track-raw-pointers" cargo +nightly miri run
   ```

2. **Custom Backend:**
   - Deploy a custom server with Rust and Miri installed
   - Update `rust_executor.py` to use your backend

### Current Implementation

The current version simulates Miri output for demonstration. To get real Miri traces:

1. Modify `services/rust_executor.py` to execute locally
2. Parse actual Miri JSON output in `services/miri_parser.py`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **Rust Playground** - For the online execution API
- **Monaco Editor** - For the code editor
- **Miri** - For Rust memory safety analysis
- **Django** - For the web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🔮 Future Enhancements

- [ ] Real Miri integration with custom backend
- [ ] Step-by-step execution
- [ ] Timeline scrubbing for memory states
- [ ] Export memory traces
- [ ] Comparison view for multiple executions
- [ ] WebSocket support for real-time updates
- [ ] Code sharing and permalinks
- [ ] Interactive memory graph with zoom/pan
- [ ] Memory leak detection
- [ ] Performance profiling

---

**Built with ❤️ for the Rust community**
