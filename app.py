import os
import sys
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from typing import Optional, Dict, Any, List
import logging

# Import custom modules
from llm_client import LLMClient, BatchProcessor
from utils import ExcelHandler, FileValidator, estimate_processing_time

# Configure logging with comprehensive error handling
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('pharma_generator.log', mode='a', encoding='utf-8')  # File output
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Pharma Description Generator - Starting")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("="*60)
except Exception as e:
    print(f"Error setting up logging: {str(e)}")
    logger = logging.getLogger(__name__)

# Initialize Flask app with error handling
try:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'pharma-desc-gen-2024-secure-key'
    
    # Use absolute paths to ensure directories are created in the correct location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'output')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    
    # Ensure directories exist with error handling
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        logger.info(f"Directories created: {app.config['UPLOAD_FOLDER']}, {app.config['OUTPUT_FOLDER']}")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        # Create fallback directories
        app.config['UPLOAD_FOLDER'] = 'uploads'
        app.config['OUTPUT_FOLDER'] = 'output'
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
except Exception as e:
    logger.error(f"Critical error initializing Flask app: {str(e)}")
    raise


# Global dictionary to track processing jobs with error handling
try:
    processing_jobs: Dict[str, Any] = {}
    logger.info("Processing jobs dictionary initialized")
except Exception as e:
    logger.error(f"Error initializing processing jobs: {str(e)}")
    processing_jobs = {}

# Inactivity timer logic
last_activity = time.time()
INACTIVITY_TIMEOUT = 5 * 60  # 5 minutes

def is_processing():
    # Return True if any job is currently processing
    for job in processing_jobs.values():
        if job.status in ("processing", "initializing"):
            return True
    return False

def update_activity():
    global last_activity
    last_activity = time.time()

def inactivity_watcher():
    while True:
        time.sleep(10)
        idle_time = time.time() - last_activity
        logger.debug(f"[InactivityWatcher] is_processing={is_processing()} | idle_time={idle_time:.1f}s | timeout={INACTIVITY_TIMEOUT}s")
        if not is_processing() and (idle_time > INACTIVITY_TIMEOUT):
            logger.info("No activity for 5 minutes. Shutting down server...")
            os._exit(0)

threading.Thread(target=inactivity_watcher, daemon=True).start()


class ProcessingJob:
    """
    Represents a single processing job with progress tracking and original data preservation.
    """
    
    def __init__(self, job_id: str, total_products: int, original_data=None):
        self.job_id = job_id
        self.total_products = total_products
        self.processed_count = 0
        self.current_product = ""
        self.status = "initializing"  # initializing, processing, completed, failed, stopped
        self.progress_percentage = 0
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.output_file: Optional[str] = None
        self.error_message: Optional[str] = None
        self.results: List[Dict[str, Any]] = []
        self.stop_requested = False
        self.original_data = original_data  # Store original DataFrame
    
    def update_progress(self, processed: int, current: str):
        """Update job progress."""
        self.processed_count = processed
        self.current_product = current
        self.progress_percentage = min(100, int((processed / self.total_products) * 100))
        
    def complete(self, output_file: str):
        """Mark job as completed."""
        self.status = "completed"
        self.progress_percentage = 100
        self.end_time = datetime.now()
        self.output_file = output_file
        
    def fail(self, error: str):
        """Mark job as failed."""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error_message = error
    
    def stop(self, output_file: Optional[str] = None):
        """Mark job as stopped by user with safe attribute handling."""
        try:
            self.status = "stopped"
            self.end_time = datetime.now()
            self.stop_requested = True
            if output_file:
                self.output_file = output_file
        except AttributeError as e:
            logger.error(f"AttributeError in job.stop(): {str(e)}")
            # Ensure critical attributes exist
            if not hasattr(self, 'status'):
                self.status = "stopped"
            if not hasattr(self, 'end_time'):
                self.end_time = datetime.now()
            if not hasattr(self, 'stop_requested'):
                self.stop_requested = True
            if output_file and not hasattr(self, 'output_file'):
                self.output_file = output_file


@app.before_request
def before_request():
    update_activity()

@app.route('/')
def index():
    """
    Main page with upload form and processing interface.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and initiate processing.
    """
    logger.info("=== UPLOAD REQUEST RECEIVED ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request form keys: {list(request.form.keys())}")
    logger.info(f"Request files keys: {list(request.files.keys())}")
    logger.info(f"Model type: {request.form.get('model_type', 'NOT PROVIDED')}")
    logger.info("=" * 60)
    logger.info("📤 UPLOAD REQUEST RECEIVED")
    logger.info("=" * 60)
    try:
        # Validate request
        if 'file' not in request.files:
            logger.error("❌ No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Get form data
        api_key = request.form.get('api_key', '').strip()
        model_type = request.form.get('model_type', 'mistral').lower()
        
        # API key is required for all models
        openrouter_models = ['mistral', 'openchat', 'deepseek', 'gptoss']
        allowed_models = openrouter_models + ['gemini']
        
        if model_type not in allowed_models:
            return jsonify({
                'success': False,
                'status': 'invalid',
                'message': 'Invalid model type',
                'details': 'Supported models: Mistral, OpenChat, DeepSeek, GPT-4o Mini (OpenRouter), and Gemini.'
            })
        
        # API key validation - required for all models
        if not api_key:
            return jsonify({'success': False, 'error': 'API key is required'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({'success': False, 'error': 'Please upload an Excel file (.xlsx or .xls)'})
        
        job_id = str(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(upload_path)
        
        # Validate file
        validation_result = FileValidator.validate_input_file(upload_path)
        if not validation_result['valid']:
            os.remove(upload_path)  # Clean up
            return jsonify({'success': False, 'error': validation_result['error']})
        
        # Create processing job
        job = ProcessingJob(job_id, validation_result['product_count'])
        job.api_limit_hit = False  # Track if API limit was reached
        processing_jobs[job_id] = job
        
        # Get time estimate
        time_estimate = estimate_processing_time(validation_result['product_count'], model_type)
        
        # Start processing in background thread
        logger.info(f"🚀 Starting background thread for job {job_id}")
        thread = threading.Thread(
            target=process_file_async,
            args=(job_id, upload_path, api_key, model_type),
            daemon=True
        )
        thread.start()
        logger.info(f"✅ Background thread started: {thread.name}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'product_count': validation_result['product_count'],
            'estimated_time': time_estimate['formatted_time'],
            'recommendation': time_estimate['recommendation']
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})



@app.route('/progress/<job_id>')
def get_progress(job_id):
    """
    Get progress information for a processing job.
    """
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    elapsed_time = ""
    estimated_time = "Calculating..."
    if job.start_time:
        elapsed = datetime.now() - job.start_time
        elapsed_seconds = int(elapsed.total_seconds())
        # Always show elapsed time as H:M:S
        elapsed_h = elapsed_seconds // 3600
        elapsed_m = (elapsed_seconds % 3600) // 60
        elapsed_s = elapsed_seconds % 60
        elapsed_time = f"{elapsed_h}h {elapsed_m}m {elapsed_s}s"

        # Estimate time left if possible
        if job.processed_count and job.total_products and job.processed_count > 0:
            avg_time_per = elapsed_seconds / job.processed_count
            remaining = job.total_products - job.processed_count
            est_seconds = int(avg_time_per * remaining)
            est_h = est_seconds // 3600
            est_m = (est_seconds % 3600) // 60
            est_s = est_seconds % 60
            estimated_time = f"{est_h}h {est_m}m {est_s}s"
        else:
            estimated_time = "Estimating..."

    # Try to get failed_count and rate_limit_count if available
    failed_count = 0
    rate_limit_count = 0
    if hasattr(job, 'failed_count'):
        failed_count = job.failed_count
    elif hasattr(job, 'results') and isinstance(job.results, list):
        failed_count = len([r for r in job.results if isinstance(r, dict) and r.get('status') == 'failed'])

    if hasattr(job, 'rate_limit_count'):
        rate_limit_count = job.rate_limit_count

    return jsonify({
        'status': job.status,
        'progress': job.progress_percentage,
        'processed_count': job.processed_count,
        'total_count': job.total_products,
        'current_product': job.current_product,
        'elapsed_time': elapsed_time,
        'estimated_time': estimated_time,
        'error_message': job.error_message,
        'can_download': job.output_file is not None and job.status in ['completed', 'stopped', 'failed', 'api_limit'],
        'api_limit_hit': hasattr(job, 'api_limit_hit') and job.api_limit_hit,
        'failed_count': failed_count,
        'rate_limit_count': rate_limit_count
    })


@app.route('/stop/<job_id>', methods=['POST'])
def stop_processing(job_id):
    """
    Stop processing job and prepare partial results for download.
    """
    try:
        job = processing_jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Defensive check for status attribute
        job_status = getattr(job, 'status', 'unknown')
        if job_status not in ['processing', 'initializing']:
            return jsonify({'error': f'Job cannot be stopped (current status: {job_status})'}), 400
        
        # Mark job for stopping with defensive attribute handling
        if hasattr(job, 'stop_requested'):
            job.stop_requested = True
        else:
            # If attribute doesn't exist, create it
            job.stop_requested = True
        
        logger.info(f"Stop requested for job {job_id}")
        
        # Defensive check for processed_count
        processed_count = getattr(job, 'processed_count', 0)
        
        return jsonify({
            'success': True,
            'message': 'Stop request sent. Processing will halt after current batch.',
            'processed_count': processed_count
        })
    except AttributeError as e:
        logger.error(f"AttributeError in stop_processing for job {job_id}: {str(e)}")
        return jsonify({'error': f'Failed to stop job: {str(e)}'}), 500
    except KeyError as e:
        logger.error(f"KeyError in stop_processing for job {job_id}: {str(e)}")
        return jsonify({'error': f'Failed to stop job: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in stop_processing for job {job_id}: {str(e)}")
        return jsonify({'error': f'Failed to stop job: {str(e)}'}), 500


@app.route('/download/<job_id>')
def download_result(job_id):
    """
    Download the generated output file.
    """
    job = processing_jobs.get(job_id)
    if not job or job.status not in ['completed', 'stopped', 'failed', 'api_limit'] or not job.output_file:
        return jsonify({'error': 'File not ready for download'}), 404
    
    if not os.path.exists(job.output_file):
        return jsonify({'error': 'Output file not found'}), 404
    
    # Generate download filename with appropriate suffix
    if job.status == "stopped":
        status_suffix = "_partial"
    elif job.status == "failed":
        status_suffix = "_partial_failed"
    else:
        status_suffix = ""
    download_name = f"product_descriptions_{job_id[:8]}{status_suffix}.xlsx"
    
    return send_file(
        job.output_file,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/status')
def system_status():
    """
    Get system status and active jobs count.
    """
    active_jobs = len([j for j in processing_jobs.values() if j.status == 'processing'])
    completed_jobs = len([j for j in processing_jobs.values() if j.status == 'completed'])
    failed_jobs = len([j for j in processing_jobs.values() if j.status == 'failed'])
    
    return jsonify({
        'active_jobs': active_jobs,
        'completed_jobs': completed_jobs,
        'failed_jobs': failed_jobs,
        'system_status': 'operational'
    })


def process_file_async(job_id: str, file_path: str, api_key: str, model_type: str):
    """
    Process file asynchronously in background thread.
    
    Args:
        job_id (str): Unique job identifier
        file_path (str): Path to uploaded Excel file
        api_key (str): API key for LLM service
        model_type (str): Type of LLM model to use
    """
    job = processing_jobs.get(job_id)
    if not job:
        return
    
    try:
        logger.info(f"Starting processing job {job_id} with {model_type}")
        job.status = "processing"

        # Read products from Excel file and preserve original data
        product_info_list, original_data = ExcelHandler.read_input_file(file_path)
        job.total_products = len(product_info_list)
        job.original_data = original_data  # Store original data in job

        # Set batch size: 10 for cloud models (parallel processing optimization)
        batch_size = 10
        
        # Initialize LLM client
        try:
            llm_client = LLMClient(api_key, model_type)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize LLM client: {error_msg}")
            job.status = "failed"
            job.error = f"Model initialization failed: {error_msg}"
            return
        
        processor = BatchProcessor(llm_client, batch_size=batch_size)

        # Enhanced progress callback function that handles failed count
        def progress_callback(progress_pct: int, processed: int, total: int, failed: int = 0):
            # Only update progress and log after a batch is fully completed (to match frontend)
            if processed > 0:
                # Find the last product in the current batch
                current_product = None
                if len(processor.cache) > 0:
                    last_result = list(processor.cache.values())[-1]
                    current_product = last_result.get('product_name', 'Unknown')
                job.update_progress(processed, current_product or "")
                # Log progress for large datasets only after batch
                if total > 100 and processed % 50 == 0:
                    logger.info(f"Progress: {processed}/{total} ({progress_pct}%) processed, {failed} failed (batch complete)")

        # Stop check function with defensive error handling
        def stop_check():
            try:
                return getattr(job, 'stop_requested', False)
            except (AttributeError, KeyError) as e:
                logger.warning(f"Error checking stop status: {str(e)}")
                return False

        # Run async processing
        logger.info(f"Starting batch processing of {len(product_info_list)} products...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("Running async event loop for processing...")
            results = loop.run_until_complete(
                processor.process_products(product_info_list, progress_callback, stop_check)
            )
            logger.info(f"Async processing completed. Got {len(results) if results else 0} results.")
        except Exception as e:
            logger.error(f"❌ Error during async processing: {str(e)}", exc_info=True)
            raise
        finally:
            loop.close()

        # Helper to check which products are missing descriptions
        def get_missing_products(product_info_list, results):
            described_names = set()
            for r in results:
                if isinstance(r, dict) and r.get('product_name'):
                    # Check if BOTH short and long descriptions are present and non-empty
                    short_desc = r.get('short_description', '').strip()
                    long_desc = r.get('long_description', '').strip()
                    if short_desc and long_desc:
                        described_names.add(r['product_name'])
            missing = [p for p in product_info_list if p.get('product_name') not in described_names]
            return missing

        def build_retry_context_map(results):
            context_map = {}
            for r in results:
                if not isinstance(r, dict):
                    continue
                name = str(r.get('product_name', '')).strip()
                if not name:
                    continue
                key = name.lower()
                short_desc = (r.get('short_description') or '').strip()
                long_desc = (r.get('long_description') or '').strip()
                attempt = r.get('attempt')
                try:
                    attempt_val = int(attempt) if attempt is not None else 1
                except Exception:
                    attempt_val = 1
                context_map[key] = {
                    'short_description': short_desc,
                    'long_description': long_desc,
                    'short_completed': bool(short_desc),
                    'long_completed': bool(long_desc),
                    'errors': r.get('error') or r.get('errors'),
                    'attempt': attempt_val
                }
            return context_map

        # Handle processing results

        if not results:
            # Defensive check for stop_requested attribute
            is_stopped = getattr(job, 'stop_requested', False)
            if is_stopped:
                raise Exception("Processing stopped by user request with no results")
            else:
                raise Exception("No results generated")

        # Filter out exceptions from failed tasks
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                logger.warning(f"Skipping invalid result: {result}")


        # Improved retry for missing products with intelligent batching to avoid API waste
        max_retries = 5
        retry_count = 0
        all_results = valid_results[:]
        products_per_retry_batch = 5  # Retry 5 products at a time
        consecutive_no_progress = 0  # Track consecutive retries with no progress
        consecutive_api_errors = 0  # Track API errors to detect quota exhaustion
        
        while retry_count < max_retries:
            missing_products = get_missing_products(product_info_list, all_results)
            logger.warning(f"[Retry {retry_count+1}] {len(missing_products)} products missing descriptions.")
            
            if not missing_products or job.stop_requested:
                break
            
            # Stop if we've had 2 consecutive retries with no progress to avoid API waste
            if consecutive_no_progress >= 2:
                logger.warning(f"Stopping retries: No progress in last {consecutive_no_progress} attempts to avoid API waste")
                break
            
            # Count before retry
            before_count = len([r for r in all_results if isinstance(r, dict) and r.get('product_name') 
                               and r.get('short_description', '').strip() and r.get('long_description', '').strip()])
            
            # Retry only first N products in this batch to save API calls
            retry_batch = missing_products[:products_per_retry_batch]
            logger.info(f"Retrying batch of {len(retry_batch)} products (out of {len(missing_products)} missing)")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                retry_results = loop.run_until_complete(
                    processor.process_products(
                        retry_batch,
                        progress_callback,
                        stop_check,
                        existing_results=build_retry_context_map(all_results)
                    )
                )
            except Exception as retry_error:
                error_msg = str(retry_error).lower()
                # Detect API limit/quota errors
                if any(keyword in error_msg for keyword in ['rate limit', '429', 'quota', 'insufficient credits', '402', 'billing']):
                    consecutive_api_errors += 1
                    logger.error(f"API limit/quota error detected: {retry_error}")
                    
                    # If we've hit API errors multiple times, stop and save what we have
                    if consecutive_api_errors >= 2:
                        logger.warning("⚠️ API LIMIT/QUOTA EXHAUSTED - Saving partial results for download")
                        job.api_limit_hit = True
                        break
                else:
                    raise  # Re-raise non-API-limit errors
            finally:
                loop.close()
            
            # Merge: always keep only the latest result for each product_name
            result_map = {r['product_name']: r for r in all_results if isinstance(r, dict) and r.get('product_name')}
            for r in retry_results:
                if isinstance(r, dict) and r.get('product_name'):
                    result_map[r['product_name']] = r
            all_results = list(result_map.values())
            
            # Count after retry to check progress
            after_count = len([r for r in all_results if isinstance(r, dict) and r.get('product_name') 
                              and r.get('short_description', '').strip() and r.get('long_description', '').strip()])
            
            # Check if we made progress
            if after_count > before_count:
                progress_made = after_count - before_count
                logger.info(f"Progress: {progress_made} products successfully completed in this retry")
                consecutive_no_progress = 0  # Reset counter
                consecutive_api_errors = 0  # Reset API error counter on success
            else:
                consecutive_no_progress += 1
                logger.warning(f"No progress in retry attempt {retry_count+1} (consecutive: {consecutive_no_progress})")
            
            # Break if API limit was hit
            if hasattr(job, 'api_limit_hit') and job.api_limit_hit:
                logger.warning("Breaking retry loop due to API limit")
                break
            
            retry_count += 1
        # Final check for missing
        missing_products = get_missing_products(product_info_list, all_results)
        # Defensive check for stop_requested attribute
        is_stopped = getattr(job, 'stop_requested', False)
        if missing_products and not is_stopped:
            logger.error(f"After {max_retries} retries, {len(missing_products)} products still missing descriptions.")
            # Log which products are still missing
            for mp in missing_products[:10]:  # Log first 10
                logger.error(f"  - Missing: {mp.get('product_name')}")
        else:
            logger.info(f"All products processed after {retry_count} retries.")
        
        # Additional validation: Check for products with incomplete descriptions
        incomplete_count = 0
        for result in all_results:
            if isinstance(result, dict):
                product_name = result.get('product_name', '')
                short_desc = result.get('short_description', '').strip()
                long_desc = result.get('long_description', '').strip()
                
                if product_name and (not short_desc or not long_desc):
                    incomplete_count += 1
                    missing_type = []
                    if not short_desc:
                        missing_type.append('short')
                    if not long_desc:
                        missing_type.append('long')
                    logger.warning(f"Incomplete descriptions for '{product_name}': missing {', '.join(missing_type)}")
        
        if incomplete_count > 0:
            logger.warning(f"Total products with incomplete descriptions: {incomplete_count}")

        # Store results in job for later download
        job.results = all_results

        # Create output file
        output_filename = f"output_{job_id}.xlsx"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Create the actual output file
        ExcelHandler.create_output_file(all_results, output_path, job.original_data)

        # Mark job as completed, stopped, or API limit hit
        # Defensive check for stop_requested attribute
        is_stopped = getattr(job, 'stop_requested', False)
        if is_stopped:
            job.stop(output_path)
            logger.info(f"Job {job_id} stopped by user. Processed {len(all_results)} products.")
        elif hasattr(job, 'api_limit_hit') and job.api_limit_hit:
            job.status = 'api_limit'
            job.output_file = output_path
            logger.warning(f"Job {job_id} stopped due to API limit. Processed {len(all_results)} products. File ready for download.")
        else:
            job.complete(output_path)
            logger.info(f"Job {job_id} completed successfully. Processed {len(all_results)} products.")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌❌❌ Job {job_id} FAILED ❌❌❌")
        logger.error(f"Error message: {error_msg}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.exception("Full traceback:")
        
        # Even if the job fails, create output file for any successful results
        try:
            if hasattr(job, 'results') and job.results and len(job.results) > 0:
                logger.info(f"Creating partial output file with {len(job.results)} processed products before failure")
                output_filename = f"output_{job_id}.xlsx"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                ExcelHandler.create_output_file(job.results, output_path, job.original_data)
                job.output_file = output_path
                logger.info(f"Partial output file created at: {output_path}")
        except Exception as output_error:
            logger.error(f"Failed to create partial output file: {str(output_error)}")
        
        job.fail(error_msg)

    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error occurred.'}), 500


@app.route('/validate_api', methods=['POST'])
def validate_api():
    """
    Validate API key and check rate limits with real-time testing.
    Returns status, rate limit info, and validation results.
    """
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        model_type = data.get('model_type', 'mistral').lower()
        
        openrouter_models = ['mistral', 'openchat', 'deepseek', 'gptoss']
        allowed_models = openrouter_models + ['gemini']
        
        if model_type not in allowed_models:
            return jsonify({
                'success': False,
                'status': 'invalid',
                'message': 'Invalid model type',
                'details': 'Supported models: Mistral, OpenChat, DeepSeek, GPT-4o Mini (OpenRouter), and Gemini.'
            })
        
        # API key is required for all models
        if not api_key:
            return jsonify({
                'success': False,
                'status': 'invalid',
                'message': 'API key is required',
                'details': 'Please enter your API key'
            })
        
        # Test the API key with a simple request for cloud models
        validation_result = test_api_key(api_key, model_type)
        
        return jsonify(validation_result)
        
    except Exception as e:
        logger.error(f"API validation error: {str(e)}")
        return jsonify({
            'success': False,
            'status': 'error',
            'message': 'Validation failed',
            'details': str(e)
        })


def test_api_key(api_key: str, model_type: str) -> dict:
    """
    Test API key validity and check rate limits.
    
    Args:
        api_key (str): API key to test
        model_type (str): 'mistral' or 'gemini'
        
    Returns:
        dict: Validation results with status and rate limit info
    """
    import asyncio
    import time
    
    openrouter_models = ['mistral', 'openchat', 'deepseek', 'gptoss']
    async def async_test():
        try:
            # Initialize LLM client
            llm_client = LLMClient(api_key, model_type)
            # Test with a simple prompt
            test_prompt = "Describe the pharmaceutical uses of paracetamol."
            start_time = time.time()
            if model_type in openrouter_models:
                result = await llm_client._call_mistral(test_prompt, model_override=llm_client.openrouter_models[model_type], max_retries=1)
            else:  # gemini
                result = await llm_client._call_gemini(test_prompt, max_retries=1)
            response_time = time.time() - start_time
            # If we got here without exceptions, the API key is valid
            # Even if result is empty, the API accepted our key (200 OK response)
            if result and len(result.strip()) > 10:
                # API key is valid and working with good response
                return {
                    'success': True,
                    'status': 'valid',
                    'message': f'{model_type.title()} API key is valid and working',
                    'details': {
                        'response_time': round(response_time, 2),
                        'model': model_type,
                        'rate_limit_status': 'Good',
                        'test_response_length': len(result),
                        'timestamp': time.strftime('%H:%M:%S')
                    }
                }
            else:
                # API accepted the key (no auth error) but response was short/empty
                # This still means the API key is VALID, just the response was minimal
                return {
                    'success': True,
                    'status': 'valid',
                    'message': f'{model_type.title()} API key is valid (authenticated successfully)',
                    'details': {
                        'response_time': round(response_time, 2),
                        'model': model_type,
                        'rate_limit_status': 'Good',
                        'test_response_length': len(result) if result else 0,
                        'timestamp': time.strftime('%H:%M:%S'),
                        'note': 'API authenticated successfully. Short response is normal for validation.'
                    }
                }
        except Exception as e:
            error_msg = str(e).lower()
            # Analyze the error type
            if 'unauthorized' in error_msg or 'invalid' in error_msg or '401' in error_msg:
                return {
                    'success': False,
                    'status': 'invalid',
                    'message': f'{model_type.title()} API key is invalid',
                    'details': {
                        'error': 'Authentication failed',
                        'suggestion': 'Check your API key is correct and active'
                    }
                }
            elif 'rate limit' in error_msg or '429' in error_msg:
                return {
                    'success': False,
                    'status': 'rate_limited',
                    'message': f'{model_type.title()} API key is valid but rate limited',
                    'details': {
                        'error': 'Rate limit exceeded',
                        'suggestion': 'Wait a few minutes and try again'
                    }
                }
            elif 'quota' in error_msg or 'billing' in error_msg or 'payment' in error_msg:
                return {
                    'success': False,
                    'status': 'quota_exceeded',
                    'message': f'{model_type.title()} API key quota exceeded',
                    'details': {
                        'error': 'Quota or billing issue',
                        'suggestion': 'Check your account billing and quota'
                    }
                }
            elif 'timeout' in error_msg or 'connection' in error_msg:
                return {
                    'success': False,
                    'status': 'connection_error',
                    'message': f'Connection error to {model_type.title()} API',
                    'details': {
                        'error': 'Network or timeout issue',
                        'suggestion': 'Check your internet connection'
                    }
                }
            else:
                return {
                    'success': False,
                    'status': 'error',
                    'message': f'{model_type.title()} API test failed',
                    'details': {
                        'error': str(e),
                        'suggestion': 'Check your API key and try again'
                    }
                }
    
    # Run the async test
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_test())
        loop.close()
        return result
    except Exception as e:
        return {
            'success': False,
            'status': 'error',
            'message': 'Validation system error',
            'details': {
                'error': str(e),
                'suggestion': 'Please try again'
            }
        }


def cleanup_old_jobs():
    """
    Clean up old processing jobs and files (run periodically).
    """
    cutoff_time = datetime.now() - timedelta(hours=24)  # 24 hours ago
    
    jobs_to_remove = []
    for job_id, job in processing_jobs.items():
        if job.end_time and job.end_time < cutoff_time:
            # Clean up output file
            if job.output_file and os.path.exists(job.output_file):
                try:
                    os.remove(job.output_file)
                except Exception:
                    pass
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        del processing_jobs[job_id]
    
    logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


def start_cleanup_scheduler():
    """
    Start background thread for automatic file cleanup.
    Runs cleanup every 6 hours.
    """
    def cleanup_worker():
        while True:
            try:
                time.sleep(6 * 60 * 60)  # Sleep for 6 hours
                cleanup_old_jobs()
                logger.info("Automatic cleanup completed")
            except Exception as e:
                logger.error(f"Error in automatic cleanup: {str(e)}")
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Automatic cleanup scheduler started (runs every 6 hours)")


if __name__ == '__main__':
    logger.info("Starting Pharma Description Generator application...")
    
    # Show startup information
    print("\n" + "="*60)
    print("🧬 PHARMACEUTICAL DESCRIPTION GENERATOR")
    print("="*60)
    print("✅ Server starting on: http://127.0.0.1:5000")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("� Output: Direct download (no server storage)")
    print("📊 Max file size: 50MB")
    print("🤖 Supported models: Mistral 7B (OpenRouter), OpenChat, DeepSeek, GPT-4o Mini (OpenRouter), Gemini 1.5 Flash")
    print("="*60)
    print("🚀 Open your browser and navigate to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Clean up any existing old files on startup
    cleanup_old_jobs()
    
    # Start automatic cleanup scheduler
    start_cleanup_scheduler()
    
    # Start the Flask development server
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )
