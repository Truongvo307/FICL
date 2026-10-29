import os
import csv
import numpy as np
from datetime import datetime


class CSVResultSaver:

    def __init__(self, args, run_time, num_clients, client_models=None):
        self.args = args
        self.run_time = run_time
        self.num_clients = num_clients
        self.client_models = client_models if client_models else {}
        self.model_family = args.model_family if hasattr(args, 'model_family') else "unknown"
        self.goal = args.goal if hasattr(args, 'goal') else "unknown"
        
        timestamp = datetime.now().strftime("%Y%m%d")
        if not hasattr(args, 'experiment_folder') or args.experiment_folder is None:
            args.experiment_folder = f"{args.dataset}_{args.algorithm}_{self.model_family}_{self.goal}_{timestamp}"
        
        self.base_result_dir = os.path.join("../results", args.experiment_folder)
        self.result_dir = os.path.join(self.base_result_dir, f"times_{run_time}")
        
        self._create_directories()
        
        self.server_results_buffer = []
        self.client_results_buffer = {i: [] for i in range(num_clients)}
        
        self.cumulative_comm_cost = 0.0
        
        self.server_headers = [
            'round',
            'global_test_acc',
            'global_test_auc',
            'global_test_loss', 
            'time_cost',
            'cumulative_time',
            'num_clients_selected',
            'best_acc_so_far',
            'comm_cost_round',
            'cumulative_comm_cost'
        ]
        
        self.client_headers = [
            'round',
            'model_name',
            'accuracy',
            'loss', 
            'auc',
            'num_samples'
        ]
    
    def _create_directories(self):
        """Create the result directory structure."""
        if not os. path.exists(self.base_result_dir):
            os.makedirs(self.base_result_dir)
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        self.clients_dir = os.path.join(self.result_dir, "clients")
        if not os.path.exists(self.clients_dir):
            os.makedirs(self.clients_dir)
    
    def get_result_dir(self):
        """Return the run-specific result directory path."""
        return self.result_dir
    
    def get_base_result_dir(self):
        """Return the base result directory path."""
        return self. base_result_dir
    
    def save_server_round_result(self, round_num, test_acc, test_auc=None, test_loss=None,
                                  time_cost=None, cumulative_time=None, num_clients=None,
                                  best_acc=None, comm_cost_round=None):
        """Buffer server-level results for a specific round."""
        if comm_cost_round is not None:
            self.cumulative_comm_cost += comm_cost_round
        
        row = [
            round_num,
            f"{test_acc:.4f}" if test_acc is not None else "",
            f"{test_auc:.4f}" if test_auc is not None else "",
            f"{test_loss:.4f}" if test_loss is not None else "",
            f"{time_cost:.2f}" if time_cost is not None else "",
            f"{cumulative_time:.2f}" if cumulative_time is not None else "",
            num_clients if num_clients is not None else "",
            f"{best_acc:.4f}" if best_acc is not None else "",
            f"{comm_cost_round:.4f}" if comm_cost_round is not None else "",
            f"{self.cumulative_comm_cost:.4f}"
        ]
        self.server_results_buffer.append(row)
    
    def save_client_round_result(self, client_id, round_num, model_name=None,
                                  accuracy=None, loss=None, auc=None, num_samples=None):
        """Buffer client-level results for a specific round."""
        if client_id not in self.client_results_buffer:
            self. client_results_buffer[client_id] = []
        
        if model_name is None:
            model_name = self. client_models.get(client_id, "")
        
        row = [
            round_num,
            model_name,
            f"{accuracy:.4f}" if accuracy is not None else "",
            f"{loss:.4f}" if loss is not None else "",
            f"{auc:.4f}" if auc is not None else "",
            num_samples if num_samples is not None else ""
        ]
        self.client_results_buffer[client_id].append(row)
    
    def flush_to_disk(self):
        """Write all buffered results to CSV files at once."""
        # Write server results
        server_csv_path = os.path.join(self.result_dir, "server.csv")
        with open(server_csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(self.server_headers)
            writer.writerows(self.server_results_buffer)
        
        # Write client results
        for client_id, rows in self.client_results_buffer.items():
            if rows:
                csv_path = os.path.join(self.clients_dir, f"client_{client_id}.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    writer.writerow(self.client_headers)
                    writer.writerows(rows)
        
        print(f"Results saved to: {self.result_dir}")
    
    def set_client_model(self, client_id, model_name):
        """Set the model name for a specific client."""
        self.client_models[client_id] = model_name


def compute_and_save_averages(base_result_dir, num_times, num_clients):
    """Compute averages across all runs and save to avg/ folder."""
    avg_dir = os.path.join(base_result_dir, "avg")
    if not os.path.exists(avg_dir):
        os.makedirs(avg_dir)
    
    avg_clients_dir = os.path.join(avg_dir, "clients")
    if not os.path. exists(avg_clients_dir):
        os.makedirs(avg_clients_dir)
    
    _compute_server_averages(base_result_dir, avg_dir, num_times)
    _compute_client_averages(base_result_dir, avg_clients_dir, num_times, num_clients)
    
    print(f"Average results saved to: {avg_dir}")


def _compute_server_averages(base_result_dir, avg_dir, num_times):
    """Compute and save server averages with std."""
    all_server_data = []
    for t in range(num_times):
        server_csv = os.path.join(base_result_dir, f"times_{t}", "server.csv")
        if os.path.exists(server_csv):
            data = _read_csv_to_dict(server_csv)
            if data:
                all_server_data.append(data)
    
    if not all_server_data:
        print("Warning: No server data found for averaging.")
        return
    
    num_rounds = min(len(d['round']) for d in all_server_data)
    numeric_cols = ['global_test_acc', 'global_test_auc', 'global_test_loss',
                    'time_cost', 'cumulative_time', 'best_acc_so_far',
                    'comm_cost_round', 'cumulative_comm_cost']
    
    avg_headers = ['round']
    for col in numeric_cols:
        avg_headers.extend([f'{col}_mean', f'{col}_std'])
    avg_headers.append('num_clients_selected')
    
    avg_rows = []
    for r in range(num_rounds):
        row = [r]
        for col in numeric_cols:
            values = []
            for data in all_server_data: 
                if r < len(data. get(col, [])) and data[col][r] != '':
                    try:
                        values.append(float(data[col][r]))
                    except ValueError:
                        pass
            
            if values:
                mean_val = np.mean(values)
                std_val = np. std(values)
                row. extend([f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                row.extend(["", ""])
        
        num_clients = all_server_data[0].get('num_clients_selected', [''])[r] if r < len(all_server_data[0]. get('num_clients_selected', [])) else ""
        row.append(num_clients)
        
        avg_rows.append(row)
    
    avg_server_csv = os.path.join(avg_dir, "server.csv")
    with open(avg_server_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(avg_headers)
        writer.writerows(avg_rows)


def _compute_client_averages(base_result_dir, avg_clients_dir, num_times, num_clients):
    """Compute and save client averages with std."""
    for client_id in range(num_clients):
        all_client_data = []
        for t in range(num_times):
            client_csv = os.path.join(base_result_dir, f"times_{t}", "clients", f"client_{client_id}.csv")
            if os.path.exists(client_csv):
                data = _read_csv_to_dict(client_csv)
                if data:
                    all_client_data.append(data)
        
        if not all_client_data: 
            continue
        
        num_rounds = min(len(d['round']) for d in all_client_data)
        numeric_cols = ['accuracy', 'loss', 'auc']
        
        avg_headers = ['round', 'model_name']
        for col in numeric_cols:
            avg_headers.extend([f'{col}_mean', f'{col}_std'])
        avg_headers.append('num_samples')
        
        avg_rows = []
        for r in range(num_rounds):
            model_name = all_client_data[0].get('model_name', [''])[r] if r < len(all_client_data[0].get('model_name', [])) else ""
            row = [r, model_name]
            
            for col in numeric_cols: 
                values = []
                for data in all_client_data: 
                    if r < len(data.get(col, [])) and data[col][r] != '':
                        try: 
                            values.append(float(data[col][r]))
                        except ValueError:
                            pass
                
                if values: 
                    mean_val = np. mean(values)
                    std_val = np.std(values)
                    row.extend([f"{mean_val:.4f}", f"{std_val:.4f}"])
                else: 
                    row.extend(["", ""])
            
            num_samples = all_client_data[0].get('num_samples', [''])[r] if r < len(all_client_data[0].get('num_samples', [])) else ""
            row. append(num_samples)
            
            avg_rows.append(row)
        
        avg_client_csv = os.path. join(avg_clients_dir, f"client_{client_id}.csv")
        with open(avg_client_csv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(avg_headers)
            writer.writerows(avg_rows)


def _read_csv_to_dict(csv_path):
    """Read CSV file and return as dictionary of lists."""
    data = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                for key, value in row.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(value)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None
    return data