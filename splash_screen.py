"""
Splash screen for MEP-CMAP Analyser
Shows while the main application loads
"""
import tkinter as tk
from tkinter import ttk

class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Splash dimensions
        splash_width = 400
        splash_height = 200
        
        # Center the splash screen
        x = (screen_width - splash_width) // 2
        y = (screen_height - splash_height) // 2
        
        self.root.geometry(f'{splash_width}x{splash_height}+{x}+{y}')
        self.root.configure(bg='#2C3E50')
        
        # Create content
        frame = tk.Frame(self.root, bg='#2C3E50')
        frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title = tk.Label(
            frame, 
            text="MEP-CMAP Analyser",
            font=('Helvetica', 18, 'bold'),
            fg='white',
            bg='#2C3E50'
        )
        title.pack(pady=(10, 20))
        
        # Loading message
        self.message = tk.Label(
            frame,
            text="Loading application...",
            font=('Helvetica', 11),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        self.message.pack(pady=10)
        
        # Progress bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Splash.Horizontal.TProgressbar",
            thickness=20,
            troughcolor='#34495E',
            background='#3498DB'
        )
        
        self.progress = ttk.Progressbar(
            frame,
            style="Splash.Horizontal.TProgressbar",
            length=300,
            mode='indeterminate'
        )
        self.progress.pack(pady=20)
        self.progress.start(10)
        
        # Version info
        version = tk.Label(
            frame,
            text="v1.0",
            font=('Helvetica', 9),
            fg='#95A5A6',
            bg='#2C3E50'
        )
        version.pack(side='bottom')
        
        self.root.update()
    
    def update_message(self, message):
        """Update the loading message"""
        self.message.config(text=message)
        self.root.update()
    
    def close(self):
        """Close the splash screen"""
        self.progress.stop()
        self.root.destroy()

def show_splash():
    """Show splash screen (called before heavy imports)"""
    splash = SplashScreen()
    return splash

if __name__ == "__main__":
    # Test the splash screen
    splash = show_splash()
    import time
    time.sleep(3)
    splash.close()
