from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .models import CodeExecution
from .services.rust_executor import RustExecutor
from .services.miri_parser import MiriParser


def index(request):
    """Main debugger interface"""
    return render(request, 'debugger/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def execute_code(request):
    """Execute Rust code with Miri and return memory trace"""
    try:
        data = json.loads(request.body)
        code = data.get('code', '')
        use_miri = data.get('use_miri', True)  # Default to Miri execution
        
        if not code:
            return JsonResponse({'error': 'No code provided'}, status=400)
        
        # Create execution record
        execution = CodeExecution.objects.create(code=code, status='running')
        
        # Execute code
        executor = RustExecutor()
        
        if use_miri:
            # Execute with Miri for memory tracing
            result = executor.execute_with_miri(code)
            
            # Parse Miri output
            parser = MiriParser()
            memory_trace = parser.parse(result.get('miri_output', ''))
            
            # Update execution record
            execution.execution_output = result.get('stdout', '')
            execution.miri_trace = memory_trace
            execution.status = 'completed' if result.get('success') else 'failed'
            execution.save()
            
            return JsonResponse({
                'execution_id': execution.id,
                'success': result.get('success', False),
                'stdout': result.get('stdout', ''),
                'stderr': result.get('stderr', ''),
                'memory_trace': memory_trace,
                'mode': 'miri'
            })
        else:
            # Execute normally without Miri
            result = executor.execute_normal(code)
            
            # Update execution record
            execution.execution_output = result.get('stdout', '')
            execution.status = 'completed' if result.get('success') else 'failed'
            execution.save()
            
            return JsonResponse({
                'execution_id': execution.id,
                'success': result.get('success', False),
                'stdout': result.get('stdout', ''),
                'stderr': result.get('stderr', ''),
                'mode': 'normal'
            })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_execution(request, execution_id):
    """Get execution details"""
    try:
        execution = CodeExecution.objects.get(id=execution_id)
        return JsonResponse({
            'id': execution.id,
            'code': execution.code,
            'output': execution.execution_output,
            'memory_trace': execution.miri_trace,
            'status': execution.status,
            'created_at': execution.created_at.isoformat(),
        })
    except CodeExecution.DoesNotExist:
        return JsonResponse({'error': 'Execution not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
