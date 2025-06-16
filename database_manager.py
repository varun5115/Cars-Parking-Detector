#!/usr/bin/env python3
"""
Database Manager for Cars Parking Detector
Handles automatic database creation, initialization, and cleanup
"""

import sqlite3
import os
import atexit
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='database/parking_data.db'):
        self.db_path = db_path
        self.connection = None
        
    def initialize_database(self):
        """Initialize the parking detection database with required tables"""
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            # Connect to database
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()
            
            print(f"🔧 Initializing database: {self.db_path}")
            
            # Create parking_spaces table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_spaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create parking_status_log table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_status_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_spaces INTEGER NOT NULL,
                occupied_spaces INTEGER NOT NULL,
                free_spaces INTEGER NOT NULL,
                occupancy_rate REAL NOT NULL,
                detection_method TEXT DEFAULT 'opencv',
                frame_number INTEGER
            )
            ''')
            
            # Create space_status_history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS space_status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id INTEGER NOT NULL,
                is_occupied BOOLEAN NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                detection_method TEXT,
                FOREIGN KEY (space_id) REFERENCES parking_spaces (space_id)
            )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_parking_status_timestamp ON parking_status_log(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_space_status_space_id ON space_status_history(space_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_space_status_timestamp ON space_status_history(timestamp)')
            
            # Commit changes
            self.connection.commit()
            
            print("✅ Database initialized successfully!")
            
            # Register cleanup function to run when program exits
            atexit.register(self.cleanup_database)
            
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def log_parking_status(self, total_spaces, occupied_spaces, detection_method='opencv', frame_number=None):
        """Log current parking status to database"""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            free_spaces = total_spaces - occupied_spaces
            occupancy_rate = (occupied_spaces / total_spaces) if total_spaces > 0 else 0.0
            
            cursor.execute('''
            INSERT INTO parking_status_log 
            (total_spaces, occupied_spaces, free_spaces, occupancy_rate, detection_method, frame_number)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (total_spaces, occupied_spaces, free_spaces, occupancy_rate, detection_method, frame_number))
            
            self.connection.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Error logging parking status: {e}")
            return False
    
    def save_parking_spaces(self, parking_spaces):
        """Save parking space coordinates to database"""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # Clear existing spaces
            cursor.execute('DELETE FROM parking_spaces')
            
            # Insert new spaces
            for i, (x, y, w, h) in enumerate(parking_spaces):
                cursor.execute('''
                INSERT INTO parking_spaces (space_id, x, y, width, height)
                VALUES (?, ?, ?, ?, ?)
                ''', (i, x, y, w, h))
            
            self.connection.commit()
            print(f"💾 Saved {len(parking_spaces)} parking spaces to database")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Error saving parking spaces: {e}")
            return False
    
    def cleanup_database(self):
        """Clean up database when program exits"""
        if self.connection:
            try:
                print("🧹 Cleaning up database...")
                
                # Optional: Keep only recent logs (last 1000 entries)
                cursor = self.connection.cursor()
                cursor.execute('''
                DELETE FROM parking_status_log 
                WHERE id NOT IN (
                    SELECT id FROM parking_status_log 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                )
                ''')
                
                # Optional: Keep only recent space history (last 24 hours)
                cursor.execute('''
                DELETE FROM space_status_history 
                WHERE timestamp < datetime('now', '-1 day')
                ''')
                
                self.connection.commit()
                self.connection.close()
                print("✅ Database cleanup completed")
                
                # Optionally delete the entire database file
                # Uncomment the next two lines if you want to delete the database completely on exit
                # if os.path.exists(self.db_path):
                #     os.remove(self.db_path)
                #     print("🗑️ Database file deleted")
                
            except Exception as e:
                print(f"❌ Error during cleanup: {e}")

# Global database manager instance
db_manager = None

def get_database_manager():
    """Get the global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        db_manager.initialize_database()
    return db_manager