import os
import logging
from datetime import datetime


# Fixed width for all separators and alignment
LOG_WIDTH = 40


def setup_logger(args, run_time, result_dir):
    """Set up a logger to save important information to a log file."""
    log_filename = f"{args.dataset}_{args.algorithm}_{args.goal}_run{run_time}.log"
    log_path = os.path.join(result_dir, log_filename)
    
    logger = logging.getLogger(f'FL_Logger_{run_time}_{datetime.now().strftime("%H%M%S")}')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def _center_title(title, width=LOG_WIDTH, char='-'):
    """Create a centered title line like:  ----- Title -----"""
    title_with_spaces = f" {title} "
    total_dashes = width - len(title_with_spaces)
    left_dashes = total_dashes // 2
    right_dashes = total_dashes - left_dashes
    return f"{char * left_dashes}{title_with_spaces}{char * right_dashes}"


def log_experiment_config(logger, args):
    """Log the experiment configuration at the start."""
    logger.info(_center_title("EXPERIMENT CONFIGURATION", char='='))
    for arg in vars(args):
        logger.info(f"{arg}:  {getattr(args, arg)}")
    logger.info("=" * (LOG_WIDTH + 11))


def log_round_results(logger, round_num, test_acc, test_auc=None, test_loss=None, time_cost=None):
    """Log results for a specific round."""
    logger.info(_center_title(f"Round {round_num} Summary"))
    logger.info(f"Test Accuracy : {test_acc:.4f}")
    if test_auc is not None: 
        logger.info(f"Test AUC      : {test_auc:.4f}")
    if test_loss is not None:
        logger.info(f"Test Loss     : {test_loss:.4f}")
    if time_cost is not None:
        logger.info(f"Time Cost     : {time_cost:.2f}s")


def log_client_results(logger, round_num, client_results, num_clients=None):
    """
    Log per-client results for a specific round with proper alignment.
    """
    if not client_results: 
        return
    
    # Determine the width needed for client ID
    if num_clients is None:
        max_client_id = max(r.get('client_id', 0) for r in client_results)
    else:
        max_client_id = num_clients - 1
    
    id_width = max(2, len(str(max_client_id)))
    
    # Column widths
    client_col = id_width + 7  # "Client " + id_width
    acc_col = 8
    loss_col = 8
    auc_col = 8
    
    logger.info(_center_title(f"Client Results for Round {round_num}"))
    
    # Header - use same format as data rows
    header_client = "Client"
    logger.info(f"{header_client:<{client_col}}  {'Acc':>{acc_col}}  {'Loss': >{loss_col}}  {'AUC':>{auc_col}}")
    logger.info(f"{'-' * client_col}  {'-' * acc_col}  {'-' * loss_col}  {'-' * auc_col}")
    
    # Sort by client_id for consistent ordering
    sorted_results = sorted(client_results, key=lambda x: x.get('client_id', 0))
    
    for result in sorted_results:
        client_id = result.get('client_id', '?')
        acc = result.get('accuracy', 0)
        loss = result.get('loss')
        auc = result.get('auc', 0)
        
        # Format client string
        client_str = f"Client {client_id: >{id_width}}"
        
        # Format values
        acc_str = f"{acc:{acc_col}.4f}"
        loss_str = f"{loss:{loss_col}.4f}" if loss is not None else f"{'N/A':>{loss_col}}"
        auc_str = f"{auc:{auc_col}.4f}"
        
        logger.info(f"{client_str: <{client_col}}  {acc_str}  {loss_str}  {auc_str}")
    
    logger.info("-" * LOG_WIDTH)


def log_final_results(logger, rs_test_acc, rs_test_loss, total_time, budget):
    """Log final results at the end of training."""
    logger.info(_center_title("FINAL RESULTS", char='='))
    if len(rs_test_acc) > 0:
        logger.info(f"Best Test Accuracy  : {max(rs_test_acc):.4f}")
        logger.info(f"Final Test Accuracy : {rs_test_acc[-1]:.4f}")
    if len(rs_test_loss) > 0:
        logger.info(f"Best Test Loss      : {min(rs_test_loss):.4f}")
        logger.info(f"Final Test Loss     : {rs_test_loss[-1]:.4f}")
    if len(budget) > 1:
        logger.info(f"Avg Time per Round  : {sum(budget[1:])/len(budget[1:]):.2f}s")
    logger.info(f"Total Training Time : {total_time:.2f}s")
    logger.info("=" * LOG_WIDTH)