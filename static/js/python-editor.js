// Python Editor initialization

let pythonEditor = null;

require.config({ 
    paths: { 
        'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' 
    }
});

require(['vs/editor/editor.main'], function() {
    pythonEditor = monaco.editor.create(document.getElementById('editor'), {
        value: `# Welcome to Python Programming!
# This editor supports Python 3 with popular libraries

import numpy as np
import pandas as pd

# Example: Create a simple array
data = np.array([1, 2, 3, 4, 5])
print(f"Array: {data}")
print(f"Mean: {data.mean()}")
print(f"Sum: {data.sum()}")
`,
        language: 'python',
        theme: 'vs-dark',
        fontSize: 14,
        minimap: { enabled: true },
        automaticLayout: true,
        scrollBeyondLastLine: false,
        wordWrap: 'on',
        tabSize: 4,
        insertSpaces: true,
        lineNumbers: 'on',
        renderWhitespace: 'selection',
        folding: true,
        bracketPairColorization: {
            enabled: true
        }
    });
    
    console.log('Python Monaco Editor initialized');
});

function getEditorCode() {
    return pythonEditor ? pythonEditor.getValue() : '';
}

function setEditorCode(code) {
    if (pythonEditor) {
        pythonEditor.setValue(code);
    }
}

function clearEditor() {
    if (pythonEditor) {
        pythonEditor.setValue('# Start coding here...\n');
    }
}
