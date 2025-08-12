from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import os
import tempfile
import uuid
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global storage for session data
sessions = {}

class DataProcessor:
    def __init__(self):
        self.df_original = None
        self.df_cleaned = None
        self.file_info = {}
        
    def load_csv(self, file_path):
        """Load CSV file with pandas and return info"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.df_original = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df_original is None:
                raise Exception("Could not read file with any encoding")
            
            # Basic info
            rows, cols = self.df_original.shape
            file_size = os.path.getsize(file_path)
            
            # Data types info
            dtypes_info = {}
            for col in self.df_original.columns:
                dtypes_info[col] = str(self.df_original[col].dtype)
            
            # Missing values info
            missing_info = {}
            for col in self.df_original.columns:
                missing_count = self.df_original[col].isnull().sum()
                missing_info[col] = int(missing_count)
            
            self.file_info = {
                'rows': rows,
                'columns': cols,
                'file_size': file_size,
                'dtypes': dtypes_info,
                'missing_values': missing_info,
                'column_names': list(self.df_original.columns)
            }
            
            return {
                'status': 'success',
                'info': self.file_info,
                'preview': self.get_preview(self.df_original, 10)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
        
    def _convert_numpy_types(self, value):
        """Convert numpy types to Python native types"""
        if pd.isna(value) or value is None:
            return None
        elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, bytes):
            return value.decode('utf-8')
        else:
            return value
    
    def clean_data(self, options):
        """Clean data based on options"""
        if self.df_original is None:
            return {'status': 'error', 'message': 'No data loaded'}
        
        try:
            self.df_cleaned = self.df_original.copy()
            original_rows = len(self.df_cleaned)
            
            cleaning_report = {
                'original_rows': original_rows,
                'steps': []
            }
            
            # Remove completely empty rows
            if options.get('remove_empty_rows', True):
                before = len(self.df_cleaned)
                self.df_cleaned = self.df_cleaned.dropna(how='all')
                after = len(self.df_cleaned)
                cleaning_report['steps'].append({
                    'step': 'Remove empty rows',
                    'removed': before - after
                })
            
            # Strip whitespace from string columns
            if options.get('strip_whitespace', True):
                string_columns = self.df_cleaned.select_dtypes(include=['object']).columns
                for col in string_columns:
                    self.df_cleaned[col] = self.df_cleaned[col].astype(str).str.strip()
                    # Convert 'nan' strings back to actual NaN
                    self.df_cleaned[col] = self.df_cleaned[col].replace('nan', np.nan)
                cleaning_report['steps'].append({
                    'step': 'Strip whitespace',
                    'affected_columns': len(string_columns)
                })
            
            # Handle missing values
            if options.get('handle_missing', True):
                missing_before = self.df_cleaned.isnull().sum().sum()
                
                for column in self.df_cleaned.columns:
                    if self.df_cleaned[column].dtype in ['object']:
                        # For string columns, fill with empty string
                        self.df_cleaned[column].fillna('', inplace=True)
                    elif self.df_cleaned[column].dtype in ['int64', 'float64']:
                        # For numeric columns, fill with median
                        median_val = self.df_cleaned[column].median()
                        self.df_cleaned[column].fillna(median_val, inplace=True)
                    else:
                        # For other types, fill with most frequent value
                        mode_val = self.df_cleaned[column].mode()
                        if len(mode_val) > 0:
                            self.df_cleaned[column].fillna(mode_val[0], inplace=True)
                
                missing_after = self.df_cleaned.isnull().sum().sum()
                cleaning_report['steps'].append({
                    'step': 'Handle missing values',
                    'filled': int(missing_before - missing_after)
                })
            
            # Remove duplicates
            if options.get('remove_duplicates', True):
                before = len(self.df_cleaned)
                self.df_cleaned = self.df_cleaned.drop_duplicates()
                after = len(self.df_cleaned)
                cleaning_report['steps'].append({
                    'step': 'Remove duplicates',
                    'removed': before - after
                })
            
            # Remove index column if it exists
            if options.get('remove_index_column', True):
                if self.df_cleaned.index.name is not None:
                    self.df_cleaned.reset_index(drop=True, inplace=True)
                    cleaning_report['steps'].append({
                    'step': 'Remove index column',
                    'removed': 'Index column removed'
                })
            
            # Data type optimization
            if options.get('optimize_dtypes', True):
                optimized_cols = []
                for col in self.df_cleaned.columns:
                    if self.df_cleaned[col].dtype == 'object':
                        try:
                            # Try to convert to numeric
                            pd.to_numeric(self.df_cleaned[col], errors='raise')
                            self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col])
                            optimized_cols.append(col)
                        except:
                            pass
                
                cleaning_report['steps'].append({
                    'step': 'Optimize data types',
                    'optimized_columns': len(optimized_cols)
                })
            
            # Remove outliers (optional)
            if options.get('remove_outliers', False):
                numeric_columns = self.df_cleaned.select_dtypes(include=[np.number]).columns
                outliers_removed = 0
                
                for col in numeric_columns:
                    Q1 = self.df_cleaned[col].quantile(0.25)
                    Q3 = self.df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    before = len(self.df_cleaned)
                    self.df_cleaned = self.df_cleaned[
                        (self.df_cleaned[col] >= lower_bound) & 
                        (self.df_cleaned[col] <= upper_bound)
                    ]
                    after = len(self.df_cleaned)
                    outliers_removed += (before - after)
                
                cleaning_report['steps'].append({
                    'step': 'Remove outliers',
                    'removed': outliers_removed
                })
            
            cleaned_rows = len(self.df_cleaned)
            cleaning_report['final_rows'] = cleaned_rows
            cleaning_report['total_removed'] = original_rows - cleaned_rows
            
            return {
                'status': 'success',
                'report': cleaning_report,
                'preview': self.get_preview(self.df_cleaned, 10),
                'info': {
                    'rows': cleaned_rows,
                    'columns': len(self.df_cleaned.columns),
                    'missing_values': {col: int(self.df_cleaned[col].isnull().sum()) 
                                     for col in self.df_cleaned.columns}
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_preview(self, df, num_rows=10):
        """Get preview of dataframe"""
        if df is None or df.empty:
            return {'columns': [], 'data': []}
        
        preview_df = df.head(num_rows)
        
        return {
            'columns': list(df.columns),
            'data': preview_df.values.tolist()
        }
    
    def save_excel(self, file_path, format_type='xlsx'):
        """Save cleaned data to Excel"""
        if self.df_cleaned is None:
            return {'status': 'error', 'message': 'No cleaned data available'}
        
        try:
            if format_type.lower() == 'xls':
            # For XLS format (Excel 97-2003)
                self.df_cleaned.to_excel(file_path, index=False, engine='xlwt')
            else:
            # For XLSX format (Excel 2007+)
                self.df_cleaned.to_excel(file_path, index=False, engine='openpyxl')
            return {'status': 'success', 'file_path': file_path}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def upload_to_database(self, db_config):
        """Upload cleaned data to PostgreSQL"""
        if self.df_cleaned is None:
            return {'status': 'error', 'message': 'No cleaned data available'}
        
        try:
            # Debug: Print connection parameters
            print(f"Connecting to database with:")
            print(f"  Host: {db_config.get('host', 'NOT SET')}")
            print(f"  Port: {db_config.get('port', 'NOT SET')}")
            print(f"  Database: {db_config.get('database', 'NOT SET')}")
            print(f"  User: {db_config.get('username', 'NOT SET')}")
            print(f"  Table: {db_config.get('table_name', 'NOT SET')}")
            
            # Validate required parameters
            required_params = ['host', 'port', 'database', 'username', 'password', 'table_name']
            missing_params = [p for p in required_params if not db_config.get(p)]
            if missing_params:
                return {'status': 'error', 'message': f'Missing required parameters: {", ".join(missing_params)}'}
            
            # Connect to database with proper error handling
            try:
                conn = psycopg2.connect(
                    host=db_config['host'],
                    port=int(db_config['port']),  # Ensure port is integer
                    database=db_config['database'],
                    user=db_config['username'],
                    password=db_config['password'],
                    connect_timeout=30
                )
            except psycopg2.OperationalError as e:
                return {'status': 'error', 'message': f'Database connection failed: {str(e)}'}
            except Exception as e:
                return {'status': 'error', 'message': f'Connection error: {str(e)}'}
            
            cur = conn.cursor()
            table_name = db_config['table_name']
            
            # Sanitize table name
            table_name = table_name.replace('-', '_').replace(' ', '_').replace('.', '_')
            if not table_name.replace('_', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').isalnum():
                return {'status': 'error', 'message': 'Invalid table name. Use only letters, numbers, and underscores.'}
            
            print(f"Creating/using table: {table_name}")
            
            # Create table if not exists with better column handling
            columns = []
            column_mapping = {}  # Map original column names to sanitized names
            
            for col in self.df_cleaned.columns:
                # Sanitize column name
                sanitized_col = str(col).replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '').replace('/', '_').replace('\\', '_')
                # Remove special characters except underscore
                sanitized_col = ''.join(c for c in sanitized_col if c.isalnum() or c == '_')
                # Ensure it doesn't start with a number
                if sanitized_col and sanitized_col[0].isdigit():
                    sanitized_col = 'col_' + sanitized_col
                if not sanitized_col:
                    sanitized_col = f'col_{len(columns)}'
                
                column_mapping[col] = sanitized_col
                
                # Determine PostgreSQL data type with better detection
                sample_data = self.df_cleaned[col].dropna()
                if len(sample_data) == 0:
                    col_type = 'TEXT'
                elif self.df_cleaned[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                    col_type = 'INTEGER'
                elif self.df_cleaned[col].dtype in ['float64', 'float32']:
                    col_type = 'NUMERIC'
                elif self.df_cleaned[col].dtype == 'bool':
                    col_type = 'BOOLEAN'
                elif pd.api.types.is_datetime64_any_dtype(self.df_cleaned[col]):
                    col_type = 'TIMESTAMP'
                else:
                    # Check if it's actually numeric data stored as string
                    try:
                        pd.to_numeric(sample_data.head(100), errors='raise')
                        col_type = 'NUMERIC'
                    except:
                        col_type = 'TEXT'
                
                columns.append(f'"{sanitized_col}" {col_type}')
                print(f"  Column: {col} -> {sanitized_col} ({col_type})")
            
            # Create table
            create_table_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})'
            print(f"Creating table with SQL: {create_table_sql}")
            
            try:
                cur.execute(create_table_sql)
                conn.commit()
                print("Table created successfully")
            except Exception as e:
                return {'status': 'error', 'message': f'Failed to create table: {str(e)}'}
            
            # Prepare data for insertion with proper handling
            print(f"Preparing {len(self.df_cleaned)} rows for insertion...")
            
            data_tuples = []
            for idx, (_, row) in enumerate(self.df_cleaned.iterrows()):
                row_data = []
                for col in self.df_cleaned.columns:
                    value = row[col]
                    converted_value = self._convert_numpy_types(value)
                    row_data.append(converted_value)
                
                data_tuples.append(tuple(row_data))
                
                # Progress logging for large datasets
                if (idx + 1) % 1000 == 0:
                    print(f"Prepared {idx + 1}/{len(self.df_cleaned)} rows")
            
            print(f"Data preparation complete. Sample row: {data_tuples[0] if data_tuples else 'No data'}")
            
            # Create column names list for INSERT
            sanitized_column_names = [f'"{column_mapping[col]}"' for col in self.df_cleaned.columns]
            
            # Insert data in batches with better error handling
            insert_sql = f'INSERT INTO "{table_name}" ({", ".join(sanitized_column_names)}) VALUES %s'
            print(f"Insert SQL: {insert_sql}")
            
            try:
                execute_values(
                    cur, 
                    insert_sql, 
                    data_tuples, 
                    page_size=1000,
                    template=None
                )
                conn.commit()
                print(f"Successfully inserted {len(data_tuples)} rows")
            except Exception as e:
                conn.rollback()
                return {'status': 'error', 'message': f'Failed to insert data: {str(e)}'}
            
            # Close connections
            cur.close()
            conn.close()
            
            return {
                'status': 'success',
                'message': f'Successfully uploaded {len(self.df_cleaned)} rows to table {table_name}',
                'rows_uploaded': len(self.df_cleaned),
                'table_name': table_name,
                'columns_created': len(columns)
            }
            
        except psycopg2.Error as e:
            return {'status': 'error', 'message': f'PostgreSQL error: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Unexpected error: {str(e)}'}

# API Routes
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Please upload a CSV file'})
    
    # Create session
    session_id = str(uuid.uuid4())
    processor = DataProcessor()
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{session_id}_{file.filename}")
    file.save(file_path)
    
    # Process file
    result = processor.load_csv(file_path)
    
    if result['status'] == 'success':
        sessions[session_id] = processor
        result['session_id'] = session_id
    
    # Clean up temp file
    os.remove(file_path)
    
    return jsonify(result)

@app.route('/clean_data', methods=['POST'])
def clean_data():
    """Clean data based on options"""
    data = request.get_json()
    session_id = data.get('session_id')
    options = data.get('options', {})
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'})
    
    processor = sessions[session_id]
    result = processor.clean_data(options)
    
    return jsonify(result)

@app.route('/download_excel', methods=['POST'])
def download_excel():
    """Download cleaned data as Excel"""
    data = request.get_json()
    session_id = data.get('session_id')
    format_type = data.get('format', 'xlsx')  # Default to xlsx
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'})
    
    processor = sessions[session_id]
    
    # Create temporary Excel file
    temp_dir = tempfile.gettempdir()
    excel_path = os.path.join(temp_dir, f"cleaned_data_{session_id}.{format_type}")
    
    result = processor.save_excel(excel_path, format_type)
    
    if result['status'] == 'success':
        return send_file(excel_path, as_attachment=True, download_name=f'cleaned_data.{format_type}')
    else:
        return jsonify(result)

@app.route('/upload_database', methods=['POST'])
def upload_database():
    """Upload cleaned data to database"""
    data = request.get_json()
    session_id = data.get('session_id')
    db_config = data.get('db_config')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'})
    
    processor = sessions[session_id]
    result = processor.upload_to_database(db_config)
    
    return jsonify(result)

@app.route('/test_connection', methods=['POST'])
def test_connection():
    """Test database connection"""
    data = request.get_json()
    db_config = data.get('db_config')
    
    try:
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['username'],
            password=db_config['password']
        )
        conn.close()
        return jsonify({'status': 'success', 'message': 'Connection successful'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_session_info', methods=['POST'])
def get_session_info():
    """Get current session information"""
    data = request.get_json()
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'})
    
    processor = sessions[session_id]
    
    info = {
        'has_original_data': processor.df_original is not None,
        'has_cleaned_data': processor.df_cleaned is not None,
        'file_info': processor.file_info
    }
    
    return jsonify({'status': 'success', 'info': info})

if __name__ == '__main__':
    print("Starting Python Data Processing Server...")
    print("Server will run on http://localhost:5000")
    print("Make sure to install dependencies: pip install flask flask-cors pandas numpy psycopg2-binary openpyxl")
    app.run(debug=True, host='0.0.0.0', port=5000)