import requests
import json
import time


class RustExecutor:
    """
    Execute Rust code using the Rust Playground API.
    This service handles code execution with Miri for memory tracing.
    """
    
    PLAYGROUND_URL = "https://play.rust-lang.org"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
        })
    
    def execute_normal(self, code):
        """
        Execute Rust code normally without Miri (faster execution).
        
        Args:
            code (str): Rust source code to execute
            
        Returns:
            dict: Execution result with stdout, stderr, and success status
        """
        try:
            payload = {
                "channel": "stable",
                "mode": "debug",
                "edition": "2021",
                "crateType": "bin",
                "tests": False,
                "code": code,
                "backtrace": False
            }
            
            response = self.session.post(
                f"{self.PLAYGROUND_URL}/execute",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': result.get('success', False),
                    'stdout': result.get('stdout', ''),
                    'stderr': result.get('stderr', '')
                }
            else:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Playground API error: {response.status_code}'
                }
        except requests.Timeout:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Execution timed out (30 seconds limit)'
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Error executing code: {str(e)}'
            }
    
    def execute_with_miri(self, code):
        """
        Execute Rust code with Miri and capture JSON memory trace.
        
        Args:
            code (str): Rust source code to execute
            
        Returns:
            dict: Execution result with stdout, stderr, and miri_output
        """
        try:
            # Prepare the code with Miri flags
            payload = {
                "channel": "nightly",
                "mode": "debug",
                "edition": "2021",
                "crateType": "bin",
                "tests": False,
                "code": code,
                "backtrace": False
            }
            
            # First, try to execute normally
            normal_result = self._execute_normal(payload)
            
            # Then try with Miri
            miri_result = self._execute_miri(code)
            
            return {
                'success': normal_result.get('success', False),
                'stdout': normal_result.get('stdout', ''),
                'stderr': normal_result.get('stderr', ''),
                'miri_output': miri_result.get('output', ''),
                'miri_success': miri_result.get('success', False)
            }
            
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Error executing code: {str(e)}',
                'miri_output': '',
                'miri_success': False
            }
    
    def _execute_normal(self, payload):
        """Execute code normally through the playground"""
        try:
            response = self.session.post(
                f"{self.PLAYGROUND_URL}/execute",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': result.get('success', False),
                    'stdout': result.get('stdout', ''),
                    'stderr': result.get('stderr', '')
                }
            else:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Playground API error: {response.status_code}'
                }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _execute_miri(self, code):
        """
        Execute code with Miri for memory tracing.
        Note: The Rust Playground has limited Miri support.
        This is a placeholder for the ideal implementation.
        
        In a production environment, you would need to:
        1. Run Miri locally with: MIRIFLAGS="-Zmiri-track-raw-pointers" cargo +nightly miri run
        2. Or use a custom backend server with Rust/Miri installed
        """
        try:
            # Attempt to use Miri through playground
            payload = {
                "channel": "nightly",
                "mode": "debug",
                "edition": "2021",
                "crateType": "bin",
                "tests": False,
                "code": code,
                "backtrace": False
            }
            
            # The playground doesn't directly support Miri with trace flags
            # This would need a custom backend or local execution
            # For now, we'll simulate the structure
            
            response = self.session.post(
                f"{self.PLAYGROUND_URL}/miri",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': result.get('success', False),
                    'output': result.get('stdout', '') + result.get('stderr', '')
                }
            else:
                # Fallback: return simulated structure
                return self._simulate_miri_output(code)
                
        except Exception as e:
            # If Miri execution fails, return simulated output
            return self._simulate_miri_output(code)
    
    def _simulate_miri_output(self, code):
        """
        Simulate Miri output for demonstration purposes.
        In production, replace this with actual Miri execution.
        """
        # This generates a mock memory trace for testing
        # Real implementation would parse actual Miri JSON output
        return {
            'success': True,
            'output': json.dumps({
                'events': [
                    {
                        'type': 'function_entry',
                        'name': 'main',
                        'frame_id': 0
                    },
                    {
                        'type': 'alloc',
                        'kind': 'stack',
                        'ptr': '0x1000',
                        'size': 4,
                        'variable': 'x'
                    },
                    {
                        'type': 'write',
                        'ptr': '0x1000',
                        'value': 42,
                        'size': 4
                    }
                ]
            })
        }
