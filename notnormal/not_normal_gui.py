# cython: infer_types=True

import copy
import csv
import traceback
import cython
import os.path
import queue
import time
import sys
import tkinter as tk
from functools import partial
from idlelib import tooltip
from os.path import basename
from threading import Thread
from tkinter import ttk, colorchooser, messagebox
import numpy as np
import pandas as pd
import pyabf
import ttkbootstrap
from PIL import Image, ImageTk
from matplotlib import patheffects as pe, rc
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from ttkbootstrap import Style
from notnormal import not_normal as nn
from notnormal.results import Events


if cython.compiled:
    COMPILED = True
    print('not_normal_gui compiled')
else:
    COMPILED = False
    print('not_normal_gui not compiled')

colours = {
    'baseline': '#FF0707',
    'threshold': '#FF9A52',
    'trace': '#5E3C99',
    'calculation_trace': '#B2ABD2',
    'filtered_trace': '#008837',
    'events': '#2dbd86'
}

PADDING = 6
WINDOW_PADDING = 3
ENTRY_WIDTH = 9
LARGE_FONT = 16
MEDIUM_FONT = 14
SMALL_FONT = 10

rc('font', size=SMALL_FONT)


class NotNormalGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Title
        self.title("Not Normal")
        # Loose focus pls
        self.bind("<FocusOut>", lambda event: self.wm_attributes('-topmost', 0))
        # Icon
        if hasattr(sys, '_MEIPASS'):
            self.image = os.path.join(sys._MEIPASS, 'data', 'landing_page.png')
        else:
            self.image = os.path.join(__file__, '..', 'data', 'landing_page.png')
        photoimage = ImageTk.PhotoImage(Image.open(self.image))
        self.iconphoto(True, photoimage)
        # Background
        self.configure(bg="black")
        # Root window size and resizability
        self.geometry("1600x900")
        # setting tkinter window size
        self.state('zoomed')
        self.resizable(True, True)
        # Dictionary to store all widgets
        self.widgets = dict()
        # Dictionary to store all windows
        self.windows = dict()
        # Figure options
        self.figure_options = dict()
        self.default_figure_options = dict()
        # Analysis options
        self.analysis_options = dict()
        self.default_analysis_options = dict()
        # Analysis results
        self.analysis_results = dict()
        self.table_columns = ['ID', 'Coordinates', 'Max Cutoff', 'Duration', 'Amplitude']
        # Trace information variables
        self.trace_information = dict()
        self.path = None
        self.trace = None
        self.time_vector = None
        self.time_step = None
        self.filtered_trace = None
        self.baseline = None
        self.threshold = None
        self.calculation_trace = None
        self.event_coordinates = None
        # Flags for ordering
        self.flags = dict(loaded=False, estimated=False, iterated=False, saved=False, running=False, plotting=False)

        # Initialise the style
        self.style = Style(theme='pulse')
        self.init_style()
        # Create layout
        self.init_layout()
        # Create key binds
        self.init_binds()
        # Create information
        self.init_load()
        # Create analysis options
        self.init_analyse()
        # Create figure
        self.init_analysis_view()
        # Create results
        self.init_results()
        # Create tooltips
        self.init_tooltips()
        # Show the landing page
        self.landing_page()

    def landing_page(self):
        # Hide all windows
        self.toggle_load('hide')
        self.toggle_analyse('hide')
        self.toggle_analysis_view('hide')
        self.toggle_analysis_view_options('hide')
        self.toggle_results('hide')

        # Landing page
        self.widgets['landing'] = dict()
        self.widgets['landing']['frame'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')
        self.widgets['landing']['frame'].grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.widgets['landing']['inner'] = ttk.Frame(self.widgets['landing']['frame'], style='primary.TFrame')
        self.widgets['landing']['frame'].columnconfigure(0, weight=1)
        self.widgets['landing']['frame'].rowconfigure(0, weight=1)
        self.widgets['landing']['inner'].grid(row=0, column=0, sticky="nsew", padx=WINDOW_PADDING,
                                              pady=WINDOW_PADDING)
        photoimage = ImageTk.PhotoImage(Image.open(self.image))
        self.widgets['landing']['image'] = tk.Label(self.widgets['landing']['inner'], image=photoimage)
        self.widgets['landing']['image'].image = photoimage
        self.widgets['landing']['button'] = ttk.Button(self.widgets['landing']['inner'], text="Browse", underline=0,
                                                       command=self.landing_load)
        self.widgets['landing']['inner'].columnconfigure(0, weight=1)
        self.widgets['landing']['inner'].rowconfigure(0, weight=1)
        self.widgets['landing']['image'].grid(row=0, column=0, sticky="nsew")
        self.widgets['landing']['button'].grid(row=1, column=0, sticky="ew")
        self.widgets['landing']['button'].focus_set()

        # Change the window size
        self.resizable(False, False)
        self.geometry("640x390")
        self.state('normal')

    def landing_load(self):
        self.load_trace()

        if self.flags['loaded']:
            # Destroy the landing page
            for key in self.widgets['landing'].keys():
                self.widgets['landing'][key].destroy()

            # Show the main window
            self.geometry("1600x900")
            self.state('zoomed')
            self.resizable(True, True)
            self.toggle_load('show')
            self.toggle_analyse('show')
            self.toggle_analysis_view('show')
            # Focus on the estimate button
            self.widgets['analyse']['estimate'].focus_set()

    def init_style(self):
        # Initialise the style
        self.style.configure('TLabel', anchor='w', font=('bold', SMALL_FONT))
        self.style.configure('TButton', font=('bold', MEDIUM_FONT), anchor='center', padding=PADDING)
        self.style.configure('TCombobox', padding=PADDING, font=('bold', SMALL_FONT))
        self.style.configure('TSpinbox', padding=PADDING, font=('bold', SMALL_FONT))
        self.style.configure('TEntry', padding=PADDING, font=('bold', SMALL_FONT))
        self.style.configure('primary.Inverse.TLabel', font=('bold', LARGE_FONT), anchor=tk.CENTER,
                             justify=tk.CENTER)
        self.style.configure('secondary.TLabelframe.Label', font=('bold', MEDIUM_FONT),
                             foreground=self.style.colors.secondary)
        self.style.configure('TNotebook', bordercolor='white', padding=0)
        self.style.configure('TNotebook.Tab', padding=0, font=MEDIUM_FONT)
        self.style.map('TNotebook.Tab', background=[('selected', self.style.colors.primary)],
                       foreground=[('selected', 'white')])
        self.style.configure('Horizontal.TFloodgauge', thickness=30, barsize=60)
        self.style.configure('Treeview.Heading', relief='flat')
        self.style.configure('Treeview.Item', indicatormargins=0, indicatorsize=0, padding=0)
        self.style.configure('Treeview.Cell', padding=0)
        self.style.configure('Treeview', fieldbackground=self.style.colors.secondary, indent=0, rowheight=10)
        self.style.configure('Small.secondary.Outline.TButton', font=('bold', SMALL_FONT))
        self.style.configure('primary.TButton')

    def init_layout(self):
        # Main window
        self.windows['main'] = ttk.Frame(self)
        self.windows['main'].pack(fill=tk.BOTH, expand=True)
        # Left window
        self.windows['left'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')
        # Center window
        self.windows['center'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')
        # Right window
        self.windows['right'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')

        # Main Layout
        self.windows['main'].rowconfigure(0, weight=1)
        self.windows['left'].grid(row=0, column=0, sticky="nsew")
        self.windows['main'].columnconfigure(1, weight=1)
        self.windows['center'].grid(row=0, column=1, sticky="nsew")
        self.windows['right'].grid(row=0, column=2, sticky="nsew")

        # Trace loading and information window
        self.windows['load'] = ttk.Frame(self.windows['left'], style='primary.TFrame')
        # Analyse window
        self.windows['analyse'] = ttk.Frame(self.windows['left'], style='primary.TFrame')
        # Analysis view window
        self.windows['analysis_view'] = ttk.Frame(self.windows['center'], style='primary.TFrame')
        # Results window
        self.windows['results'] = ttk.Frame(self.windows['right'], style='primary.TFrame')

        # Left layout
        self.windows['left'].columnconfigure(0, weight=1, minsize=300)
        self.windows['left'].rowconfigure(0, weight=1)
        self.windows['load'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0), padx=(WINDOW_PADDING, 0))
        self.windows['left'].rowconfigure(1, weight=2)
        self.windows['analyse'].grid(row=1, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=(WINDOW_PADDING, 0))
        # Center layout
        self.windows['center'].columnconfigure(0, weight=1)
        self.windows['center'].rowconfigure(0, weight=1)
        self.windows['analysis_view'].grid(row=0, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=WINDOW_PADDING)
        # Right layout
        self.windows['right'].columnconfigure(0, weight=1, minsize=300)
        self.windows['right'].rowconfigure(0, weight=1)
        self.windows['results'].grid(row=0, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=(0, WINDOW_PADDING))

    def init_binds(self):
        self.bind('<Control-b>', lambda event: self.load_trace())
        self.bind('<Control-e>', lambda event: self.initial_estimate())
        self.bind('<Control-i>', lambda event: self.iterate())
        self.bind('<Control-s>', lambda event: self.save_results())
        self.bind('<o>', lambda event: self.toggle_analysis_view_options())

    def init_tooltips(self):
        tooltip.Hovertip(self.widgets['load']['browse'], "Ctrl + B")
        tooltip.Hovertip(self.widgets['analyse']['estimate'], "Ctrl + E")
        tooltip.Hovertip(self.widgets['analyse']['iterate'], "Ctrl + I")
        tooltip.Hovertip(self.widgets['results']['save'], "Ctrl + S")
        tooltip.Hovertip(self.widgets['analysis_view']['options_toggle'], "O")
        tooltip.Hovertip(self.widgets['analyse']['bounds_input'],
                         "Median filtering window size for bounding events")
        tooltip.Hovertip(self.widgets['analyse']['estimate_cutoff_input'], 'Cutoff frequency for the estimate')
        tooltip.Hovertip(self.widgets['analyse']['threshold_window_input'], 'Window size for threshold calculation')
        tooltip.Hovertip(self.widgets['analyse']['z_score_input'], 'Z-score for threshold')
        tooltip.Hovertip(self.widgets['analyse']['cutoff_input'], 'Cutoff frequency for the iteration')
        tooltip.Hovertip(self.widgets['analyse']['event_direction_input'], 'Initial iteration event direction')
        tooltip.Hovertip(self.widgets['analyse']['replace_factor_input'], 'Factor for replacing events')
        tooltip.Hovertip(self.widgets['analyse']['replace_gap_input'], 'Gap for replacing events')

    def init_load(self):
        self.widgets['load'] = dict()
        # Load title
        self.widgets['load']['title'] = ttk.Label(self.windows['load'], text="Load",
                                                  style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['load_internal'] = ttk.Frame(self.windows['load'])
        # Layout
        self.windows['load'].columnconfigure(0, weight=1)
        self.widgets['load']['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        self.windows['load'].rowconfigure(1, weight=1)
        self.windows['load_internal'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

        # Filename information
        self.widgets['load']['filename_label'] = ttk.Label(self.windows['load_internal'], text="Filename")
        self.trace_information['filename'] = tk.StringVar()
        self.trace_information['filename'].set('')
        self.widgets['load']['filename'] = ttk.Label(self.windows['load_internal'],
                                                     textvariable=self.trace_information['filename'], wraplength=160)
        self.widgets['load']['filename_separator'] = ttk.Separator(self.windows['load_internal'])
        # Sample rate information
        self.widgets['load']['sample_rate_label'] = ttk.Label(self.windows['load_internal'], text="Sample Rate (Hz)")
        self.trace_information['sample_rate'] = tk.StringVar()
        self.trace_information['sample_rate'].set('')
        self.widgets['load']['sample_rate'] = ttk.Label(self.windows['load_internal'],
                                                        textvariable=self.trace_information['sample_rate'])
        self.widgets['load']['sample_rate_separator'] = ttk.Separator(self.windows['load_internal'])
        # Samples information
        self.widgets['load']['samples_label'] = ttk.Label(self.windows['load_internal'], text="Samples")
        self.trace_information['samples'] = tk.StringVar()
        self.trace_information['samples'].set('')
        self.widgets['load']['samples'] = ttk.Label(self.windows['load_internal'],
                                                    textvariable=self.trace_information['samples'])
        self.widgets['load']['samples_separator'] = ttk.Separator(self.windows['load_internal'])
        # Duration information
        self.widgets['load']['duration_label'] = ttk.Label(self.windows['load_internal'], text="Duration (s)")
        self.trace_information['duration'] = tk.StringVar()
        self.trace_information['duration'].set('')
        self.widgets['load']['duration'] = ttk.Label(self.windows['load_internal'],
                                                     textvariable=self.trace_information['duration'])
        # Browse button
        self.widgets['load']['browse'] = ttk.Button(self.windows['load_internal'], text="Browse", underline=0,
                                                    command=self.load_trace, style='secondary.Outline.TButton')

        # Filename layout
        self.windows['load_internal'].columnconfigure(0, weight=1)
        self.windows['load_internal'].columnconfigure(1, weight=2)
        self.windows['load_internal'].rowconfigure(0, weight=1)
        self.widgets['load']['filename_label'].grid(row=0, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['filename'].grid(row=0, column=1, sticky="nsew", padx=PADDING)
        self.widgets['load']['filename_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        # Sample rate layout
        self.windows['load_internal'].rowconfigure(2, weight=1)
        self.widgets['load']['sample_rate_label'].grid(row=2, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['sample_rate'].grid(row=2, column=1, sticky="nsew", padx=PADDING)
        self.widgets['load']['sample_rate_separator'].grid(row=3, column=0, columnspan=2, sticky="nsew")
        # Samples layout
        self.windows['load_internal'].rowconfigure(4, weight=1)
        self.widgets['load']['samples_label'].grid(row=4, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['samples'].grid(row=4, column=1, sticky="nsew", padx=PADDING)
        self.widgets['load']['samples_separator'].grid(row=5, column=0, columnspan=2, sticky="nsew")
        # Duration layout
        self.windows['load_internal'].rowconfigure(6, weight=1)
        self.widgets['load']['duration_label'].grid(row=6, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['duration'].grid(row=6, column=1, sticky="nsew", padx=6)
        # Browse layout
        self.widgets['load']['browse'].grid(row=7, column=0, columnspan=2, sticky="nsew", padx=PADDING,
                                            pady=(0, PADDING))

    def init_analyse(self):
        self.widgets['analyse'] = dict()
        # Options title
        self.widgets['analyse']['title'] = ttk.Label(self.windows['analyse'], text="Analyse",
                                                     style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['analyse_internal'] = ttk.Frame(self.windows['analyse'])
        # Layout
        self.windows['analyse'].columnconfigure(0, weight=1)
        self.widgets['analyse']['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        self.windows['analyse'].rowconfigure(1, weight=1)
        self.windows['analyse_internal'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

        # Bounds filtering input
        self.analysis_options['bounds_filter'] = tk.IntVar()
        self.analysis_options['bounds_filter'].set(3)
        self.widgets['analyse']['bounds_label'] = ttk.Label(self.windows['analyse_internal'],
                                                            text="Bounds Filter (Samples)")
        self.widgets['analyse']['bounds_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=13,
            increment=1,
            textvariable=self.analysis_options['bounds_filter'],
            format='%d',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['bounds_filter'] = self.analysis_options['bounds_filter'].get()
        self.widgets['analyse']['bounds_separator'] = ttk.Separator(self.windows['analyse_internal'])
        # Estimate cutoff input
        self.analysis_options['estimate_cutoff'] = tk.DoubleVar()
        self.analysis_options['estimate_cutoff'].set(10.0)
        self.widgets['analyse']['estimate_cutoff_label'] = ttk.Label(self.windows['analyse_internal'],
                                                                     text="Estimate Cutoff (Hz)")
        self.widgets['analyse']['estimate_cutoff_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=1000,
            increment=1,
            textvariable=self.analysis_options['estimate_cutoff'],
            format='%4.2f',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['estimate_cutoff'] = self.analysis_options['estimate_cutoff'].get()
        self.widgets['analyse']['estimate_cutoff_separator'] = ttk.Separator(self.windows['analyse_internal'])
        # Threshold window input
        self.analysis_options['threshold_window'] = tk.DoubleVar()
        self.analysis_options['threshold_window'].set(2.0)
        self.widgets['analyse']['threshold_window_label'] = ttk.Label(self.windows['analyse_internal'],
                                                                      text="Threshold Window (s)")
        self.widgets['analyse']['threshold_window_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=10,
            increment=0.1,
            textvariable=self.analysis_options['threshold_window'],
            format='%3.1f',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['threshold_window'] = self.analysis_options['threshold_window'].get()
        self.widgets['analyse']['threshold_window_separator'] = ttk.Separator(self.windows['analyse_internal'])
        # Z-score input
        self.analysis_options['z_score'] = tk.DoubleVar()
        self.analysis_options['z_score'].set(4.0)
        self.widgets['analyse']['z_score_label'] = ttk.Label(self.windows['analyse_internal'], text="Z-score")
        self.widgets['analyse']['z_score_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=10,
            increment=0.1,
            textvariable=self.analysis_options['z_score'],
            format='%4.3f',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['z_score'] = self.analysis_options['z_score'].get()
        # Estimate button
        self.widgets['analyse']['estimate'] = ttk.Button(
            self.windows['analyse_internal'],
            text="Estimate",
            underline=0,
            command=self.initial_estimate,
            style='secondary.Outline.TButton'
        )
        # Cutoff input
        self.analysis_options['cutoff'] = tk.IntVar()
        self.analysis_options['cutoff'].set(10.0)
        self.widgets['analyse']['cutoff_label'] = ttk.Label(self.windows['analyse_internal'], text="Cutoff (Hz)")
        self.widgets['analyse']['cutoff_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=1000,
            increment=1,
            textvariable=self.analysis_options['cutoff'],
            format='%4.2f',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['cutoff'] = self.analysis_options['cutoff'].get()
        self.widgets['analyse']['cutoff_separator'] = ttk.Separator(self.windows['analyse_internal'])
        # Starting direction input
        self.analysis_options['event_direction'] = tk.StringVar()
        self.analysis_options['event_direction'].set('down')
        self.widgets['analyse']['event_direction_label'] = ttk.Label(self.windows['analyse_internal'],
                                                                     text="Event Direction")
        self.widgets['analyse']['event_direction_input'] = ttk.Combobox(
            self.windows['analyse_internal'],
            values=['down', 'up', 'biphasic'],
            textvariable=self.analysis_options['event_direction'],
            state='readonly',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['event_direction'] = self.analysis_options['event_direction'].get()
        self.widgets['analyse']['event_direction_separator'] = ttk.Separator(self.windows['analyse_internal'])
        # Replace factor input
        self.analysis_options['replace_factor'] = tk.IntVar()
        self.analysis_options['replace_factor'].set(8)
        self.widgets['analyse']['replace_factor_label'] = ttk.Label(self.windows['analyse_internal'],
                                                                    text="Replace Factor")
        self.widgets['analyse']['replace_factor_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=16,
            increment=1,
            textvariable=self.analysis_options['replace_factor'],
            format='%d',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['replace_factor'] = self.analysis_options['replace_factor'].get()
        self.widgets['analyse']['replace_factor_separator'] = ttk.Separator(self.windows['analyse_internal'])
        # Replace gap input
        self.analysis_options['replace_gap'] = tk.IntVar()
        self.analysis_options['replace_gap'].set(2)
        self.widgets['analyse']['replace_gap_label'] = ttk.Label(self.windows['analyse_internal'], text="Replace Gap")
        self.widgets['analyse']['replace_gap_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=4,
            increment=1,
            textvariable=self.analysis_options['replace_gap'],
            format='%d',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['replace_gap'] = self.analysis_options['replace_gap'].get()
        # Run button
        self.widgets['analyse']['iterate'] = ttk.Button(
            self.windows['analyse_internal'],
            text="Iterate",
            underline=0,
            command=self.iterate,
            style='secondary.Outline.TButton'
        )

        # Bounds layout
        self.windows['analyse_internal'].columnconfigure(0, weight=1)
        self.windows['analyse_internal'].columnconfigure(1, weight=2)
        self.windows['analyse_internal'].rowconfigure(0, weight=1)
        self.widgets['analyse']['bounds_label'].grid(row=0, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['bounds_input'].grid(row=0, column=1, sticky="e", padx=PADDING)
        self.widgets['analyse']['bounds_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        # Estimate cutoff layout
        self.windows['analyse_internal'].rowconfigure(2, weight=1)
        self.widgets['analyse']['estimate_cutoff_label'].grid(row=2, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['estimate_cutoff_input'].grid(row=2, column=1, sticky="e", padx=PADDING)
        self.widgets['analyse']['estimate_cutoff_separator'].grid(row=3, column=0, columnspan=2, sticky="nsew")
        # Threshold window layout
        self.windows['analyse_internal'].rowconfigure(4, weight=1)
        self.widgets['analyse']['threshold_window_label'].grid(row=4, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['threshold_window_input'].grid(row=4, column=1, sticky="e", padx=PADDING)
        self.widgets['analyse']['threshold_window_separator'].grid(row=5, column=0, columnspan=2, sticky="nsew")
        # Z-score layout
        self.windows['analyse_internal'].rowconfigure(6, weight=1)
        self.widgets['analyse']['z_score_label'].grid(row=6, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['z_score_input'].grid(row=6, column=1, sticky="e", padx=PADDING)
        # Estimate button layout
        self.widgets['analyse']['estimate'].grid(row=7, column=0, columnspan=2, sticky="nsew", padx=PADDING)
        # Cutoff layout
        self.windows['analyse_internal'].rowconfigure(8, weight=1)
        self.widgets['analyse']['cutoff_label'].grid(row=8, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['cutoff_input'].grid(row=8, column=1, sticky="e", padx=PADDING)
        self.widgets['analyse']['cutoff_separator'].grid(row=9, column=0, columnspan=2, sticky="nsew")
        # Starting direction layout
        self.windows['analyse_internal'].rowconfigure(10, weight=1)
        self.widgets['analyse']['event_direction_label'].grid(row=10, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['event_direction_input'].grid(row=10, column=1, sticky="e", padx=PADDING)
        self.widgets['analyse']['event_direction_separator'].grid(row=11, column=0, columnspan=2, sticky="nsew")
        # Replace factor layout
        self.windows['analyse_internal'].rowconfigure(12, weight=1)
        self.widgets['analyse']['replace_factor_label'].grid(row=12, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['replace_factor_input'].grid(row=12, column=1, sticky="e", padx=PADDING)
        self.widgets['analyse']['replace_factor_separator'].grid(row=13, column=0, columnspan=2, sticky="nsew")
        # Replace gap layout
        self.windows['analyse_internal'].rowconfigure(14, weight=1)
        self.widgets['analyse']['replace_gap_label'].grid(row=14, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analyse']['replace_gap_input'].grid(row=14, column=1, sticky="e", padx=PADDING)
        # Iterate button layout
        self.widgets['analyse']['iterate'].grid(row=15, column=0, columnspan=2, sticky="nsew", pady=(0, PADDING),
                                                padx=PADDING)

    def init_analysis_view(self):
        self.widgets['analysis_view'] = dict()
        # Analysis view title
        self.widgets['analysis_view']['title'] = ttk.Label(self.windows['analysis_view'], text="Analysis View",
                                                           style='primary.Inverse.TLabel', anchor='center')
        # Show/hide options button
        self.widgets['analysis_view']['options_toggle'] = ttk.Button(
            self.windows['analysis_view'],
            text="Options",
            underline=0,
            command=self.toggle_analysis_view_options,
            style='primary.TButton'
        )
        # Internal frame for figure
        self.windows['analysis_view_figure'] = ttk.Frame(self.windows['analysis_view'])
        # Internal frame for options
        self.windows['analysis_view_options'] = ttk.Frame(self.windows['analysis_view'])
        # Layout
        self.windows['analysis_view'].columnconfigure(0, weight=1)
        self.widgets['analysis_view']['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        self.windows['analysis_view'].rowconfigure(1, weight=2)
        self.windows['analysis_view_figure'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING,
                                                  pady=WINDOW_PADDING)
        self.widgets['analysis_view']['options_toggle'].grid(row=2, column=0, sticky="ew")
        self.windows['analysis_view'].rowconfigure(3, weight=1)
        self.windows['analysis_view_options'].grid(row=3, column=0, sticky="nsew", padx=WINDOW_PADDING,
                                                   pady=(0, WINDOW_PADDING))

        # Create figure, axis and canvas
        self.widgets['analysis_view']['fig'] = Figure(layout='constrained')
        ax = self.widgets['analysis_view']['fig'].add_subplot(111)
        ax.set_xlabel("Time (s)", fontsize=MEDIUM_FONT)
        ax.set_ylabel("Current", fontsize=MEDIUM_FONT)
        self.widgets['analysis_view']['canvas'] = FigureCanvasTkAgg(self.widgets['analysis_view']['fig'],
                                                                    self.windows['analysis_view_figure'])
        self.widgets['analysis_view']['canvas'].draw()
        # Create toolbar
        self.widgets['analysis_view']['toolbar'] = NavigationToolbar2Tk(self.widgets['analysis_view']['canvas'],
                                                                        self.windows['analysis_view_figure'],
                                                                        pack_toolbar=False)
        for children in self.widgets['analysis_view']['toolbar'].winfo_children():
            children.configure(background='white')
        self.widgets['analysis_view']['toolbar'].update()
        # Progress bar
        self.widgets['analysis_view']['progress_bar'] = ttkbootstrap.Floodgauge(
            self.windows['analysis_view_figure'],
            orient=tk.HORIZONTAL,
            mode='indeterminate',
            length=180,
            style='Horizontal.TFloodgauge'
        )
        # Toolbar separator
        self.widgets['analysis_view']['toolbar_separator'] = ttk.Separator(self.windows['analysis_view_figure'])
        # Reset labelframe
        self.widgets['analysis_view']['reset'] = ttk.LabelFrame(self.windows['analysis_view_options'], text="Reset",
                                                                style='secondary.TLabelframe', labelanchor='n')
        # General labelframe
        self.widgets['analysis_view']['general'] = ttk.LabelFrame(self.windows['analysis_view_options'], text="General",
                                                                  style='secondary.TLabelframe', labelanchor='n')
        # Lines labelframe
        self.widgets['analysis_view']['lines'] = ttk.LabelFrame(self.windows['analysis_view_options'], text="Lines",
                                                                style='secondary.TLabelframe', labelanchor='n')
        # General table options
        self.widgets['analysis_view']['reset_label'] = ttk.Label(self.widgets['analysis_view']['general'],
                                                                 text="Reset")

        # Reset all
        self.widgets['analysis_view']['reset_all_button'] = ttk.Button(
            self.widgets['analysis_view']['reset'],
            text="All",
            width=2 * ENTRY_WIDTH,
            command=self.reset_all,
            style='Small.secondary.Outline.TButton'
        )
        # Reset analyse
        self.widgets['analysis_view']['reset_algorithm_button'] = ttk.Button(
            self.widgets['analysis_view']['reset'],
            text="Algorithm Options",
            width=2 * ENTRY_WIDTH,
            command=self.reset_algorithm,
            style='Small.secondary.Outline.TButton'
        )
        # Reset analysis view
        self.widgets['analysis_view']['reset_figure_button'] = ttk.Button(
            self.widgets['analysis_view']['reset'],
            text="Figure Options",
            width=2 * ENTRY_WIDTH,
            command=self.reset_figure,
            style='Small.secondary.Outline.TButton'
        )
        # Reset results
        self.widgets['analysis_view']['reset_results_button'] = ttk.Button(
            self.widgets['analysis_view']['reset'],
            text="Results",
            width=2 * ENTRY_WIDTH,
            command=self.reset_results,
            style='Small.secondary.Outline.TButton'
        )
        # Parallel option
        self.widgets['analysis_view']['parallel_label'] = ttk.Label(self.widgets['analysis_view']['general'],
                                                                    text="Parallel Compute")
        self.analysis_options['parallel'] = tk.BooleanVar()
        self.analysis_options['parallel'].set(True)
        self.widgets['analysis_view']['parallel'] = ttk.Checkbutton(
            self.widgets['analysis_view']['general'],
            variable=self.analysis_options['parallel'],
            onvalue=True,
            offvalue=False,
            style='Roundtoggle.Toolbutton',
        )
        self.default_analysis_options['parallel'] = self.analysis_options['parallel'].get()
        self.widgets['analysis_view']['parallel_separator'] = ttk.Separator(self.widgets['analysis_view']['general'])
        # X-Limit
        self.widgets['analysis_view']['x_limit_frame'] = ttk.Frame(self.widgets['analysis_view']['general'])
        self.widgets['analysis_view']['x_limit_label'] = ttk.Label(self.widgets['analysis_view']['general'],
                                                                   text="X-Limit (s)")
        self.figure_options['x_limit_lower'] = tk.DoubleVar()
        self.figure_options['x_limit_lower'].set(0.0)
        self.widgets['analysis_view']['x_limit_lower'] = ttk.Entry(
            self.widgets['analysis_view']['x_limit_frame'],
            textvariable=self.figure_options['x_limit_lower'],
            width=ENTRY_WIDTH + 1,
            validate='all',
            validatecommand=(self.register(self.validate_x_limit), '%V', '%P'),
            justify='center'
        )
        self.widgets['analysis_view']['x_limit_lower'].bind(
            '<Return>',
            partial(self.validate_x_limit, '<Return>')
        )
        self.default_figure_options['x_limit_lower'] = self.figure_options['x_limit_lower'].get()

        self.figure_options['x_limit_upper'] = tk.DoubleVar()
        self.figure_options['x_limit_upper'].set(1.0)
        self.widgets['analysis_view']['x_limit_upper'] = ttk.Entry(
            self.widgets['analysis_view']['x_limit_frame'],
            textvariable=self.figure_options['x_limit_upper'],
            width=ENTRY_WIDTH + 1,
            validate='all',
            validatecommand=(self.register(self.validate_x_limit), '%V', '%P'),
            justify='center'
        )
        self.widgets['analysis_view']['x_limit_upper'].bind(
            '<Return>',
            partial(self.validate_x_limit, '<Return>')
        )
        self.default_figure_options['x_limit_upper'] = self.figure_options['x_limit_upper'].get()
        self.widgets['analysis_view']['x_limit_separator'] = ttk.Separator(self.widgets['analysis_view']['general'])
        # Grid
        self.widgets['analysis_view']['grid_label'] = ttk.Label(self.widgets['analysis_view']['general'],
                                                                text="Grid")
        self.figure_options['grid'] = tk.BooleanVar()
        self.figure_options['grid'].set(False)
        self.widgets['analysis_view']['grid'] = ttk.Checkbutton(
            self.widgets['analysis_view']['general'],
            variable=self.figure_options['grid'],
            onvalue=True,
            offvalue=False,
            command=lambda: (self.widgets['analysis_view']['fig'].axes[0].grid(self.figure_options['grid'].get()),
                             self.widgets['analysis_view']['fig'].canvas.draw()),
            style='Roundtoggle.Toolbutton',
        )
        self.default_figure_options['grid'] = self.figure_options['grid'].get()
        self.widgets['analysis_view']['grid_separator'] = ttk.Separator(self.widgets['analysis_view']['general'])
        # Lines headings
        self.widgets['analysis_view']['show_label'] = ttk.Label(self.widgets['analysis_view']['lines'], text="Show",
                                                                anchor='center')
        self.widgets['analysis_view']['linewidth_label'] = ttk.Label(self.widgets['analysis_view']['lines'],
                                                                     text="Linewidth", anchor='center')
        self.widgets['analysis_view']['colour_label'] = ttk.Label(self.widgets['analysis_view']['lines'], text="Colour",
                                                                  anchor='center')
        self.widgets['analysis_view']['style_label'] = ttk.Label(self.widgets['analysis_view']['lines'], text="Style",
                                                                 anchor='center')
        self.widgets['analysis_view']['heading_separator'] = ttk.Separator(self.widgets['analysis_view']['lines'])
        # Lines rows
        self.widgets['analysis_view']['trace_label'] = ttk.Label(self.widgets['analysis_view']['lines'], text="Trace")
        self.widgets['analysis_view']['trace_separator'] = ttk.Separator(self.widgets['analysis_view']['lines'])
        self.widgets['analysis_view']['baseline_label'] = ttk.Label(self.widgets['analysis_view']['lines'],
                                                                    text="Baseline")
        self.widgets['analysis_view']['baseline_separator'] = ttk.Separator(self.widgets['analysis_view']['lines'])
        self.widgets['analysis_view']['threshold_label'] = ttk.Label(self.widgets['analysis_view']['lines'],
                                                                     text="Threshold")
        self.widgets['analysis_view']['threshold_separator'] = ttk.Separator(self.widgets['analysis_view']['lines'])
        self.widgets['analysis_view']['calculation_trace_label'] = ttk.Label(self.widgets['analysis_view']['lines'],
                                                                             text="Calculation Trace")
        self.widgets['analysis_view']['calculation_trace_separator'] = ttk.Separator(
            self.widgets['analysis_view']['lines'])
        self.widgets['analysis_view']['filtered_trace_label'] = ttk.Label(self.widgets['analysis_view']['lines'],
                                                                          text="Filtered Trace")
        self.widgets['analysis_view']['filtered_trace_separator'] = ttk.Separator(
            self.widgets['analysis_view']['lines'])
        self.widgets['analysis_view']['events_label'] = ttk.Label(self.widgets['analysis_view']['lines'], text="Events")

        # Lines table options
        keys = ['trace', 'baseline', 'threshold', 'calculation_trace', 'filtered_trace', 'events']
        self.figure_options['show'] = dict()
        self.default_figure_options['show'] = dict()
        self.figure_options['linewidth'] = dict()
        self.default_figure_options['linewidth'] = dict()
        self.figure_options['colour'] = dict()
        self.default_figure_options['colour'] = dict()
        self.figure_options['style'] = dict()
        self.default_figure_options['style'] = dict()
        self.widgets['analysis_view']['show'] = dict()
        self.widgets['analysis_view']['linewidth'] = dict()
        self.widgets['analysis_view']['colour'] = dict()
        self.widgets['analysis_view']['style'] = dict()
        for key in keys:
            # Show
            self.figure_options['show'][key] = tk.BooleanVar()
            self.figure_options['show'][key].set(True if key in ['trace', 'baseline', 'threshold'] else False)
            self.widgets['analysis_view']['show'][key] = ttk.Checkbutton(
                self.widgets['analysis_view']['lines'],
                variable=self.figure_options['show'][key],
                onvalue=True,
                offvalue=False,
                command=partial(
                    self.update_figure_show,
                    key
                ),
                style='Roundtoggle.Toolbutton',
            )
            self.default_figure_options['show'][key] = self.figure_options['show'][key].get()
            # Linewidth
            self.figure_options['linewidth'][key] = tk.DoubleVar()
            self.figure_options['linewidth'][key].set(1.0)
            self.widgets['analysis_view']['linewidth'][key] = ttk.Spinbox(
                self.widgets['analysis_view']['lines'],
                from_=0,
                to=4,
                increment=0.1,
                textvariable=self.figure_options['linewidth'][key],
                format='%2.1f',
                command=partial(
                    self.update_figure_linewidth,
                    key,
                    ''
                ),
                justify='center',
                width=ENTRY_WIDTH
            )
            self.widgets['analysis_view']['linewidth'][key].bind(
                "<Return>",
                partial(
                    self.update_figure_linewidth,
                    key
                )
            )
            self.default_figure_options['linewidth'][key] = self.figure_options['linewidth'][key].get()
            # Colour
            self.figure_options['colour'][key] = tk.StringVar()
            self.figure_options['colour'][key].set(colours[key])
            self.widgets['analysis_view']['colour'][key] = ttk.Button(
                self.widgets['analysis_view']['lines'],
                command=partial(
                    self.update_figure_color,
                    key
                ),
                width=ENTRY_WIDTH - 2,
            )
            self.style.configure(
                f'{key}_colour.TButton',
                background=self.figure_options['colour'][key].get(),
                bordercolor=self.figure_options['colour'][key].get(),
                lightcolor=self.figure_options['colour'][key].get(),
                darkcolor=self.figure_options['colour'][key].get(),
                padding=2,
            )
            self.style.map(
                f'{key}_colour.TButton',
                background=[('active', self.figure_options['colour'][key].get())],
                bordercolor=[('active', self.figure_options['colour'][key].get())],
                lightcolor=[('active', self.figure_options['colour'][key].get())],
                darkcolor=[('active', self.figure_options['colour'][key].get())]
            )
            self.widgets['analysis_view']['colour'][key].configure(style=f'{key}_colour.TButton')
            self.default_figure_options['colour'][key] = self.figure_options['colour'][key].get()
            # Style
            self.figure_options['style'][key] = tk.StringVar()
            self.figure_options['style'][key].set('-')
            self.widgets['analysis_view']['style'][key] = ttk.Combobox(
                self.widgets['analysis_view']['lines'],
                values=['-', '--', ':', '-.'],
                textvariable=self.figure_options['style'][key],
                state='readonly',
                justify='center',
                width=ENTRY_WIDTH
            )
            self.widgets['analysis_view']['style'][key].bind(
                '<<ComboboxSelected>>',
                partial(
                    self.update_figure_style,
                    key
                )
            )
            self.default_figure_options['style'][key] = self.figure_options['style'][key].get()

        # Figure frame layout
        self.windows['analysis_view_figure'].columnconfigure(0, weight=2)
        self.widgets['analysis_view']['toolbar'].grid(row=0, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['progress_bar'].grid(row=0, column=1, sticky='nse', padx=PADDING, pady=PADDING)
        self.widgets['analysis_view']['toolbar_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.windows['analysis_view_figure'].rowconfigure(2, weight=1)
        self.widgets['analysis_view']['canvas'].get_tk_widget().grid(row=2, column=0,  columnspan=2, sticky="nsew",
                                                                     padx=2 * PADDING, pady=2 * PADDING)

        # Options frame layout
        self.windows['analysis_view_options'].rowconfigure(0, weight=1)
        self.windows['analysis_view_options'].columnconfigure(0, weight=1)
        self.widgets['analysis_view']['reset'].grid(row=0, column=0, sticky="nsew", padx=PADDING, pady=PADDING)
        self.windows['analysis_view_options'].columnconfigure(1, weight=1)
        self.widgets['analysis_view']['general'].grid(row=0, column=1, sticky="nsew", padx=0, pady=PADDING)
        self.windows['analysis_view_options'].columnconfigure(2, weight=1)
        self.widgets['analysis_view']['lines'].grid(row=0, column=2, sticky="nsew", padx=PADDING, pady=PADDING)
        # Reset layout
        self.widgets['analysis_view']['reset'].columnconfigure(0, weight=1, uniform='options')
        self.widgets['analysis_view']['reset'].rowconfigure(0, weight=1, uniform='options')
        self.widgets['analysis_view']['reset_algorithm_button'].grid(row=0, column=0, padx=PADDING, pady=(PADDING, 0))
        self.widgets['analysis_view']['reset'].rowconfigure(1, weight=1, uniform='options')
        self.widgets['analysis_view']['reset_figure_button'].grid(row=1, column=0, padx=PADDING)
        self.widgets['analysis_view']['reset'].rowconfigure(2, weight=1, uniform='options')
        self.widgets['analysis_view']['reset_results_button'].grid(row=2, column=0, padx=PADDING)
        self.widgets['analysis_view']['reset'].rowconfigure(3, weight=1, uniform='options')
        self.widgets['analysis_view']['reset_all_button'].grid(row=3, column=0, padx=PADDING, pady=(0, PADDING))

        # General table options layout
        # Parallel layout
        self.widgets['analysis_view']['general'].columnconfigure(0, weight=1, uniform='options')
        self.widgets['analysis_view']['general'].columnconfigure(1, weight=1, uniform='options')
        self.widgets['analysis_view']['general'].rowconfigure(0, weight=1, uniform='options')
        self.widgets['analysis_view']['parallel_label'].grid(row=0, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['parallel'].grid(row=0, column=1, padx=PADDING)
        self.widgets['analysis_view']['parallel_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        # Grid
        self.widgets['analysis_view']['general'].rowconfigure(2, weight=1, uniform='options')
        self.widgets['analysis_view']['grid_label'].grid(row=2, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['grid'].grid(row=2, column=1, padx=PADDING)
        self.widgets['analysis_view']['grid_separator'].grid(row=3, column=0, columnspan=2, sticky="nsew")
        # X limit
        self.widgets['analysis_view']['x_limit_lower'].pack(side='left', expand=True)
        self.widgets['analysis_view']['x_limit_upper'].pack(side='right', expand=True)
        self.widgets['analysis_view']['general'].rowconfigure(4, weight=1, uniform='options')
        self.widgets['analysis_view']['x_limit_label'].grid(row=4, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['x_limit_frame'].grid(row=4, column=1, padx=PADDING)
        self.widgets['analysis_view']['x_limit_separator'].grid(row=5, column=0, columnspan=2, sticky="nsew")

        # Lines headings layout
        self.widgets['analysis_view']['lines'].columnconfigure(1, weight=1, uniform='options')
        self.widgets['analysis_view']['lines'].rowconfigure(0, weight=1, uniform='options')
        self.widgets['analysis_view']['show_label'].grid(row=0, column=1, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['lines'].columnconfigure(2, weight=1, uniform='options')
        self.widgets['analysis_view']['linewidth_label'].grid(row=0, column=2, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['lines'].columnconfigure(3, weight=1, uniform='options')
        self.widgets['analysis_view']['colour_label'].grid(row=0, column=3, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['lines'].columnconfigure(4, weight=1, uniform='options')
        self.widgets['analysis_view']['style_label'].grid(row=0, column=4, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['lines'].columnconfigure(0, weight=1, uniform='options')
        self.widgets['analysis_view']['heading_separator'].grid(row=1, column=0, columnspan=5, sticky="nsew")
        # Lines rows layout
        self.widgets['analysis_view']['lines'].rowconfigure(2, weight=1, uniform='options')
        self.widgets['analysis_view']['trace_label'].grid(row=2, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['trace_separator'].grid(row=3, column=0, columnspan=5, sticky="nsew")
        self.widgets['analysis_view']['lines'].rowconfigure(4, weight=1, uniform='options')
        self.widgets['analysis_view']['baseline_label'].grid(row=4, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['baseline_separator'].grid(row=5, column=0, columnspan=5, sticky="nsew")
        self.widgets['analysis_view']['lines'].rowconfigure(6, weight=1, uniform='options')
        self.widgets['analysis_view']['threshold_label'].grid(row=6, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['threshold_separator'].grid(row=7, column=0, columnspan=5, sticky="nsew")
        self.widgets['analysis_view']['lines'].rowconfigure(8, weight=1, uniform='options')
        self.widgets['analysis_view']['calculation_trace_label'].grid(row=8, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['calculation_trace_separator'].grid(row=9, column=0, columnspan=5, sticky="nsew")
        self.widgets['analysis_view']['lines'].rowconfigure(10, weight=1, uniform='options')
        self.widgets['analysis_view']['filtered_trace_label'].grid(row=10, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['filtered_trace_separator'].grid(row=11, column=0, columnspan=5, sticky="nsew")
        self.widgets['analysis_view']['lines'].rowconfigure(12, weight=1, uniform='options')
        self.widgets['analysis_view']['events_label'].grid(row=12, column=0, sticky="nsew", padx=PADDING)
        # Lines table options layout
        for row, key in enumerate(keys):
            self.widgets['analysis_view']['show'][key].grid(row=(row + 1) * 2, column=1, padx=PADDING)
            self.widgets['analysis_view']['linewidth'][key].grid(row=(row + 1) * 2, column=2, padx=PADDING)
            self.widgets['analysis_view']['colour'][key].grid(row=(row + 1) * 2, column=3, padx=PADDING)
            self.widgets['analysis_view']['style'][key].grid(row=(row + 1) * 2, column=4, padx=PADDING)

        # Bind key press event
        self.widgets['analysis_view']['canvas'].mpl_connect("key_press_event",
                                                            partial(self.toolbar_key_press, 'analysis_view'))

    def init_results(self):
        self.widgets['results'] = dict()
        # Results title
        self.widgets['results']['title'] = ttk.Label(self.windows['results'], text="Results",
                                                     style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['results_internal'] = ttk.Frame(self.windows['results'])
        # Layout
        self.windows['results'].columnconfigure(0, weight=1)
        self.widgets['results']['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        self.windows['results'].rowconfigure(1, weight=1)
        self.windows['results_internal'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

        # Results notebook
        self.widgets['results']['notebook'] = ttk.Notebook(self.windows['results_internal'])
        self.widgets['results']['notebook'].enable_traversal()
        # Results save button
        self.widgets['results']['save'] = ttk.Button(
            self.windows['results_internal'],
            text="Save",
            underline=0,
            command=self.save_results,
            style='secondary.Outline.TButton'
        )

        # Frame layout
        self.windows['results_internal'].columnconfigure(0, weight=1)
        self.windows['results_internal'].rowconfigure(0, weight=1)
        self.widgets['results']['notebook'].grid(row=0, column=0, sticky="nsew")
        self.widgets['results']['save'].grid(row=1, column=0, sticky="nsew", padx=PADDING, pady=PADDING)

    def reset_load(self):
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        # Reset the trace vectors
        self.path = None
        self.trace = None
        self.time_vector = None
        self.time_step = None
        # Clear the trace information
        for key in self.trace_information:
            self.trace_information[key].set('')
        # Set the flags
        self.flags['loaded'] = False

    def reset_algorithm(self):
        for key in self.default_analysis_options:
            self.analysis_options[key].set(self.default_analysis_options[key])

    def reset_figure(self):
        # Reset the figure options
        for key in self.default_figure_options:
            if isinstance(self.default_figure_options[key], dict):
                for sub_key in self.default_figure_options[key]:
                    self.figure_options[key][sub_key].set(self.default_figure_options[key][sub_key])
            else:
                self.figure_options[key].set(self.default_figure_options[key])
        # Update the figure
        self.set_x_limits()
        self.update_figure(retain_view=False)

    def reset_results(self):
        # Reset the trace vectors
        self.filtered_trace = None
        self.baseline = None
        self.threshold = None
        self.calculation_trace = None
        self.event_coordinates = None
        # Clear the results
        self.analysis_results = dict()
        # Update the results
        self.update_results()
        # Update the figure
        self.update_figure(title=' ')
        # Set the flags
        self.flags['estimated'] = False
        self.flags['iterated'] = False
        self.flags['saved'] = False

    def reset_all(self):
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        # Reset the load window
        self.reset_load()
        # Reset the 'analyse' window
        self.reset_algorithm()
        # Reset the results window
        self.reset_results()
        # Reset the analysis view window
        self.reset_figure()

    def update_load(self):
        if self.path is not None:
            if len(basename(self.path)) > 63:
                self.trace_information['filename'].set(f'{basename(self.path)[0:60]}...')
            else:
                self.trace_information['filename'].set(basename(self.path))

        if self.trace is not None:
            self.trace_information['samples'].set(f'{len(self.trace):.0f}')

        if self.time_vector is not None:
            self.trace_information['sample_rate'].set(f'{1 / self.time_step:.0f}')

        if self.time_vector is not None:
            self.trace_information['duration'].set(f'{self.time_vector[-1] - self.time_vector[0]:.1f}')

    @cython.boundscheck(False)
    def update_figure(self, title=None, retain_view=True):
        if self.widgets['analysis_view']['fig'].axes:
            ax = self.widgets['analysis_view']['fig'].axes[0]
        else:
            return

        if type(self.time_vector) is np.ndarray and self.time_step:
            # Limits
            lower = int(np.round((self.figure_options['x_limit_lower'].get() - self.time_vector[0]) / self.time_step))
            upper = int(np.round((self.figure_options['x_limit_upper'].get() - self.time_vector[0]) / self.time_step))
            # Clear lines except the trace unless limits changed
            replot_trace = True
            for lines in ax.get_lines():
                if lines.get_label() == 'trace':
                    xlower = int(np.round((lines.get_xdata()[0] - self.time_vector[0]) / self.time_step))
                    xupper = int(np.round((lines.get_xdata()[-1] - self.time_vector[0]) / self.time_step))
                    if lower != xlower or upper != xupper:
                        lines.remove()
                    else:
                        replot_trace = False
                else:
                    lines.remove()

            time_vector = self.time_vector[lower:upper + 1]
            # Plot the trace
            if type(self.trace) is np.ndarray and replot_trace:
                ax.plot(time_vector, self.trace[lower:upper + 1], label='trace')

            # Plot the calculation trace
            if type(self.calculation_trace) is np.ndarray:
                ax.plot(time_vector, self.calculation_trace[lower:upper + 1],
                        label='calculation_trace')

            # Plot the filtered trace
            if type(self.filtered_trace) is np.ndarray:
                ax.plot(time_vector, self.filtered_trace[lower:upper + 1],
                        label='filtered_trace')

            # Plot the baseline
            if type(self.baseline) is np.ndarray:
                ax.plot(time_vector, self.baseline[lower:upper + 1], label='baseline')

                # Plot the threshold
                if type(self.threshold) is np.ndarray:
                    ax.plot(time_vector, self.baseline[lower:upper + 1] + self.threshold[lower:upper + 1],
                            label='threshold')
                    ax.plot(time_vector, self.baseline[lower:upper + 1] - self.threshold[lower:upper + 1],
                            label='threshold')

            # Plot the events
            xlist = []
            ylist = []
            if type(self.event_coordinates) is np.ndarray:
                for event in self.event_coordinates:
                    if event[0] < lower or event[1] > upper:
                        continue
                    xlist.extend(self.time_vector[event[0]:event[1] + 1])
                    xlist.append(None)
                    ylist.extend(self.trace[event[0]:event[1] + 1])
                    ylist.append(None)
                ax.plot(xlist, ylist, label='events')

            # Set visibility, colour, linewidth and linestyle
            for line in ax.get_lines():
                if line.get_label() not in colours.keys():
                    continue
                line.set_visible(self.figure_options['show'][line.get_label()].get())
                line.set_color(self.figure_options['colour'][line.get_label()].get())
                line.set_linewidth(self.figure_options['linewidth'][line.get_label()].get())
                line.set_linestyle(self.figure_options['style'][line.get_label()].get())
        else:
            ax.clear()

        # Set limits
        if not retain_view:
            ax.relim()
            ax.autoscale(True)
            self.widgets['analysis_view']['toolbar'].update()
            self.widgets['analysis_view']['toolbar'].push_current()

        # Set title
        if title:
            ax.set_title(title, fontsize=LARGE_FONT)

        # Set grid
        ax.grid(self.figure_options['grid'].get())
        self.widgets['analysis_view']['canvas'].draw()

    def update_figure_show(self, key):
        if not self.widgets['analysis_view']['fig'].axes:
            return

        show = self.figure_options['show'][key].get()
        lines = self.widgets['analysis_view']['fig'].axes[0].get_lines()
        for line in lines:
            if key == line.get_label():
                if show == line.get_visible():
                    continue
                line.set_visible(show)
        self.widgets['analysis_view']['fig'].canvas.draw()

    def update_figure_color(self, key):
        colour = colorchooser.askcolor(title="Choose color")[1]
        if colour is None:
            return

        self.figure_options['colour'][key].set(colour)
        self.style.configure(
            f'{key}_colour.TButton',
            background=colour,
            bordercolor=colour,
            lightcolor=colour,
            darkcolor=colour
        )
        self.style.map(
            f'{key}_colour.TButton',
            background=[('active', colour)],
            bordercolor=[('active', colour)],
            lightcolor=[('active', colour)],
            darkcolor=[('active', colour)]
        )
        self.widgets['analysis_view']['colour'][key].configure(style=f'{key}_colour.TButton')

        if not self.widgets['analysis_view']['fig'].axes or not self.figure_options['show'][key].get():
            return

        lines = self.widgets['analysis_view']['fig'].axes[0].get_lines()
        for line in lines:
            if key == line.get_label():
                if colour == line.get_color():
                    continue
                line.set_color(colour)
        self.widgets['analysis_view']['fig'].canvas.draw()

    def update_figure_linewidth(self, key, event):
        if not self.widgets['analysis_view']['fig'].axes or not self.figure_options['show'][key].get():
            return

        width = self.figure_options['linewidth'][key].get()
        lines = self.widgets['analysis_view']['fig'].axes[0].get_lines()
        for line in lines:
            if key == line.get_label():
                if width == line.get_linewidth():
                    continue
                line.set_linewidth(width)
        self.widgets['analysis_view']['fig'].canvas.draw()

    def update_figure_style(self, key, event):
        if not self.widgets['analysis_view']['fig'].axes or not self.figure_options['show'][key].get():
            return

        style = self.figure_options['style'][key].get()
        lines = self.widgets['analysis_view']['fig'].axes[0].get_lines()
        for line in lines:
            if key == line.get_label():
                if style == line.get_linestyle():
                    continue
                line.set_linestyle(style)
        self.widgets['analysis_view']['fig'].canvas.draw()

    def update_results(self):
        # Add the final tab
        tabs = []
        if 'Final' in self.analysis_results.keys():
            tabs.append(self.create_tab(self.analysis_results['Final'], 'Final'))

        # Add the iteration tab
        if 'Iteration' in self.analysis_results.keys():
            tabs.append(self.create_iteration_tab())

        # Add the estimate tab
        if 'Estimate' in self.analysis_results.keys():
            tabs.append(self.create_tab(self.analysis_results['Estimate'], 'Estimate'))

        # Destroy the old, add the new
        for widget in self.widgets['results']['notebook'].winfo_children():
            if str(widget) in self.widgets['results']['notebook'].tabs():
                self.widgets['results']['notebook'].forget(widget)
                widget.destroy()

        for tab in tabs:
            self.widgets['results']['notebook'].add(tab[0], text=tab[1])

        if self.widgets['results']['notebook'].tabs():
            self.widgets['results']['notebook'].select(0)
            self.toggle_results('show')
        else:
            self.toggle_results('hide')

    def toggle_load(self, toggle=None):
        if toggle is None and self.windows['load'].winfo_manager():
            toggle = 'hide'
        elif toggle is None:
            toggle = 'show'

        if self.windows['load'].winfo_manager() and toggle == 'hide':
            self.windows['left'].grid_forget()
            self.windows['load'].grid_forget()
        elif toggle == 'show':
            self.windows['load'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0),
                                      padx=(WINDOW_PADDING, 0))
            self.windows['left'].grid(row=0, column=0, sticky="nsew")

    def toggle_analyse(self, toggle=None):
        if toggle is None and self.windows['analyse'].winfo_manager():
            toggle = 'hide'
        elif toggle is None:
            toggle = 'show'

        if self.windows['analyse'].winfo_manager() and toggle == 'hide':
            self.windows['left'].grid_forget()
            self.windows['analyse'].grid_forget()
        elif toggle == 'show':
            self.windows['analyse'].grid(row=1, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=(WINDOW_PADDING, 0))
            self.windows['left'].grid(row=0, column=0, sticky="nsew")

    def toggle_analysis_view(self, toggle=None):
        if toggle is None and self.windows['analysis_view'].winfo_manager():
            toggle = 'hide'
        elif toggle is None:
            toggle = 'show'

        if self.windows['analysis_view'].winfo_manager() and toggle == 'hide':
            self.windows['center'].grid_forget()
            self.windows['analysis_view'].grid_forget()
        elif toggle == 'show':
            self.windows['analysis_view'].grid(row=0, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=WINDOW_PADDING)
            self.windows['center'].grid(row=0, column=1, sticky="nsew")

    def toggle_analysis_view_options(self, toggle=None):
        if toggle is None and self.windows['analysis_view_options'].winfo_manager():
            toggle = 'hide'
        elif toggle is None:
            toggle = 'show'

        if self.windows['analysis_view_options'].winfo_manager() and toggle == 'hide':
            self.windows['analysis_view'].rowconfigure(3, weight=0)
            self.windows['analysis_view_options'].grid_forget()
        elif toggle == 'show':
            self.windows['analysis_view'].rowconfigure(3, weight=1)
            self.windows['analysis_view_options'].grid(row=3, column=0, sticky="nsew", padx=WINDOW_PADDING,
                                                       pady=(0, WINDOW_PADDING))

    def toggle_results(self, toggle=None):
        if toggle is None and self.windows['results'].winfo_manager():
            toggle = 'hide'
        elif toggle is None:
            toggle = 'show'

        if self.windows['results'].winfo_manager() and toggle == 'hide':
            self.windows['right'].grid_forget()
            self.windows['results'].grid_forget()
        elif toggle == 'show':
            self.windows['results'].grid(row=0, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=(0, WINDOW_PADDING))
            self.windows['right'].grid(row=0, column=2, sticky="nsew")

    def create_tab(self, results, title):
        # Create tab and frames
        tab = ttk.Frame(self.widgets['results']['notebook'])
        header_separator = ttk.Separator(tab, style='primary.TSeparator')
        stats_frame = ttk.Frame(tab)
        event_frame = ttk.Frame(tab)
        footer_separator = ttk.Separator(tab, style='primary.TSeparator')

        # Tab layout
        tab.columnconfigure(0, weight=1)
        header_separator.grid(row=0, column=0, columnspan=2, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        stats_frame.grid(row=1, column=0, sticky="nsew", pady=(0, PADDING))
        tab.rowconfigure(2, weight=5)
        event_frame.grid(row=2, column=0, sticky="nsew")
        footer_separator.grid(row=3, column=0, columnspan=2, sticky="nsew")

        # Populate the main statistics
        main_stats = {**results.trace_stats, **results.event_stats}
        labels = []
        values = []
        seperators = []
        stats_frame.columnconfigure(0, weight=2)
        stats_frame.columnconfigure(1, weight=1)
        row = 0
        for i, stat in enumerate(main_stats):
            stats_frame.rowconfigure(row, weight=1)
            labels.append(ttk.Label(stats_frame, text=stat))
            labels[-1].grid(row=row, column=0, sticky="nsew", padx=PADDING)
            values.append(ttk.Label(stats_frame, text=f'{main_stats[stat]:.6g}'))
            # stats = main_stats[stat]
            # main_stats[stat] = tk.DoubleVar()
            # main_stats[stat].set(stats)
            # values.append(ttk.Label(stats_frame, textvariable=main_stats[stat]))
            values[-1].grid(row=row, column=1, sticky="nsw", padx=PADDING)
            if i != len(main_stats) - 1:
                seperators.append(ttk.Separator(stats_frame))
                seperators[-1].grid(row=row + 1, column=0, columnspan=2, sticky="nsew")
            row += 2

        # Populate the event statistics
        event_stats = results.events.events
        if not event_stats:
            return tab, title

        keys = [key for key in event_stats[0].keys()]
        table_stats = [{} for i in range(len(event_stats))]
        max_width = dict()
        # Format
        for key in keys:
            if key not in self.table_columns:
                continue

            for i, stat in enumerate(event_stats):
                if key == 'Coordinates':
                    table_stats[i][key] = f'({stat[key][0]}, {stat[key][1]})'
                else:
                    table_stats[i][key] = f'{np.round(stat[key], 2):g}'
            row_width = max([len(stat[key]) for stat in table_stats]) * 9
            max_width[key] = row_width if row_width > len(str(key)) * 8 else len(str(key)) * 8
        for i, stat in enumerate(event_stats):
            table_stats[i]['Outlier'] = stat['Outlier']

        # Event table
        tree = ttk.Treeview(
            event_frame,
            columns=[key for key in keys if key in self.table_columns],
            style='primary.Treeview',
            padding=0,
            show='headings',
        )
        # Tags for colours
        tree.tag_configure('outlier', background=colours['baseline'], foreground=self.style.colors.secondary)
        tree.tag_configure('not_outlier', background='white', foreground=self.style.colors.secondary)

        # Set the headings
        for key in keys:
            if key not in self.table_columns:
                continue

            tree.heading(key, anchor='center', text=key, command=lambda _col=key:
                         self.sort_column(tree, _col, False))
            if key == 'ID':
                tree.column(key, width=max_width[key], anchor='w', stretch=0)
            else:
                tree.column(key, width=max_width[key], anchor='center', stretch=0)

        # Populate the table
        for i in range(len(table_stats)):
            table_data = ()
            for key in table_stats[i].keys():
                if key not in self.table_columns:
                    continue

                table_data += (table_stats[i][key],)

            tree.insert("", tk.END, text=f'{i + 1}', values=table_data,
                        tags=('outlier' if table_stats[i]['Outlier'] else 'not_outlier'))

        # Binds
        tree.bind("<Double-1>", partial(self.jump_to_event, title))
        tree.bind("<Return>", partial(self.jump_to_event, title))
        tree.bind("<space>", partial(self.jump_to_event, title))
        tree.bind("<Button-3>", partial(self.outlier_event, title))
        tree.bind("<Delete>", partial(self.outlier_event, title))

        # Scrollbar
        scrollbar = ttk.Scrollbar(event_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # Table layout
        event_frame.rowconfigure(0, weight=1)
        scrollbar.grid(row=0, column=0, sticky="ns")
        event_frame.columnconfigure(1, weight=1)
        tree.grid(row=0, column=1, sticky="nsew", pady=(0, 1))

        return tab, title

    def create_iteration_tab(self):
        if 'Iteration' in self.analysis_results.keys():
            trace_stats = self.analysis_results['Iteration']['trace_stats']
            event_stats = self.analysis_results['Iteration']['event_stats']
        else:
            return

        # Create tab and frames
        tab = ttk.Frame(self.widgets['results']['notebook'])
        header_separator = ttk.Separator(tab, style='primary.TSeparator')
        graph_frame = ttk.Frame(tab)

        # Tab layout
        tab.columnconfigure(0, weight=1)
        header_separator.grid(row=0, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        graph_frame.grid(row=1, column=0, sticky="nsew", pady=PADDING, padx=(0, PADDING))
        footer_separator = ttk.Separator(tab, style='primary.TSeparator')
        footer_separator.grid(row=2, column=0, sticky="nsew")

        # Graph frame
        self.update()
        notebook_width = (self.widgets['results']['notebook'].winfo_reqwidth() - 6 * PADDING) / 100
        length = len(trace_stats[0].keys()) + len(event_stats[0].keys())
        fig = Figure(layout='constrained', dpi=100, figsize=(notebook_width, 1.5 * length))
        # Trace stats
        x = np.arange(1, len(trace_stats) + 1)
        for i, key in enumerate(trace_stats[0].keys()):
            ax = fig.add_subplot(length, 1, i + 1)
            ax.axvline(x=len(trace_stats), color=colours['baseline'], linestyle='--', linewidth=0.5)
            ax.plot(x, [stats[key] for stats in trace_stats], label=key, color=colours['trace'], linewidth=0.5,
                    path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
            ax.scatter(x, [stats[key] for stats in trace_stats], label=key, color=colours['threshold'],
                       path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
            ax.set_title(key, fontsize=SMALL_FONT)
            ax.tick_params(axis='both', which='both', labelsize=SMALL_FONT - 2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Event stats
        x = np.arange(1, len(event_stats) + 1)
        for i, key in enumerate(event_stats[0].keys()):
            ax = fig.add_subplot(length, 1, i + 1 + len(trace_stats[0].keys()))
            ax.axvline(x=len(trace_stats), color=colours['baseline'], linestyle='--', linewidth=0.5)
            ax.plot(x, [stats[key] for stats in event_stats], label=key, color=colours['trace'], linewidth=0.5,
                    path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
            ax.scatter(x, [stats[key] for stats in event_stats], label=key, color=colours['threshold'],
                       path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
            ax.set_title(key, fontsize=SMALL_FONT)
            ax.tick_params(axis='both', which='both', labelsize=SMALL_FONT - 2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.supxlabel('Iteration', fontsize=SMALL_FONT)
        # Margins and share x ticks
        for axis in fig.get_axes():
            axis.margins(x=0.1, y=0.1)
        for ax in fig.get_axes()[1:]:
            ax.sharex(fig.get_axes()[0])

        # Inner canvas and scrollbar
        inner_canvas = tk.Canvas(graph_frame)
        graph_scrollbar = ttk.Scrollbar(graph_frame, orient="vertical")
        # Configure the canvas size and scrollbar
        size = fig.get_size_inches() * fig.dpi
        inner_canvas.configure(width=size[0])
        inner_canvas.config(yscrollcommand=graph_scrollbar.set, scrollregion=(0, 0, 0, size[1]),
                            yscrollincrement=size[1] / (length * 10))
        graph_scrollbar.config(command=inner_canvas.yview)
        # Graph frame layout
        graph_frame.rowconfigure(0, weight=1)
        graph_scrollbar.grid(row=0, column=0, sticky="ns")
        graph_frame.columnconfigure(1, weight=1)
        inner_canvas.grid(row=0, column=1, sticky="nsew", padx=PADDING)

        # Inner frame inside inner canvas
        inner_frame = ttk.Frame(inner_canvas)
        inner_canvas.create_window(0, 0, window=inner_frame, anchor=tk.NW)
        # Plot inside inner frame... inside inner canvas
        canvas = FigureCanvasTkAgg(fig, inner_frame)
        # Create toolbar
        toolbar = NavigationToolbar2Tk(canvas, inner_frame)
        for children in toolbar.winfo_children():
            children.configure(background='white')
        toolbar.update()
        toolbar.push_current()
        toolbar.pack(side=tk.TOP, fill=tk.X, expand=1)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        return tab, 'Iteration'

    def sort_column(self, tree, col, reverse):
        def check_column(value):
            try:
                return float(value[0])
            except:
                return float(eval(value[0])[0])

        l = [(tree.set(k, col), k) for k in tree.get_children('')]
        l.sort(reverse=reverse, key=check_column)

        for index, (val, k) in enumerate(l):
            tree.move(k, '', index)

        tree.heading(col, command=lambda: self.sort_column(tree, col, not reverse))

    def outlier_event(self, title, event):
        # Get the event ID
        tree = event.widget
        if not tree.selection():
            return
        item = tree.selection()[0]
        event_id = int(tree.item(item, 'text'))

        events = self.analysis_results[title].events
        # Toggle the outlier flag
        if events.get(event_id)['Outlier']:
            events.get(event_id)['Outlier'] = False
            tree.item(item, tags='not_outlier')
        else:
            events.get(event_id)['Outlier'] = True
            tree.item(item, tags='outlier')

        if title == 'Final':
            return

        event_coordinates = []
        for event in events.events:
            if not event['Outlier'] and event['Direction'] == self.analysis_options['event_direction'].get():
                event_coordinates.append(event['Coordinates'])

        # Recalculate cutoff
        self.analysis_results[title].event_stats['Max Cutoff'] = np.round(
            np.percentile(
                nn.calculate_cutoffs(
                    self.trace,
                    self.baseline,
                    self.threshold,
                    np.asarray(event_coordinates),
                    self.analysis_options['estimate_cutoff'].get(),
                    int(1 / self.time_step)
                ),
                25
            ),
            2
        )

        self.analysis_options['cutoff'].set(self.analysis_results[title].event_stats['Max Cutoff'])
        self.flash_entry(self.widgets['analyse']['cutoff_input'])

    def jump_to_event(self, title, event):
        # Get the event ID
        tree = event.widget
        if not tree.selection():
            return
        item = tree.selection()[0]
        event_id = int(tree.item(item, 'text'))

        events = self.analysis_results[title].events
        # Get the event coordinates
        event_coordinates = events.get(event_id)['Coordinates']
        extra = (event_coordinates[1] - event_coordinates[0]) * 20
        start = event_coordinates[0] - extra if event_coordinates[0] - extra > 0 else 0
        end = event_coordinates[1] + extra if event_coordinates[1] + extra < len(self.trace) else len(self.trace) - 1

        lower = int(np.round((self.figure_options['x_limit_lower'].get() - self.time_vector[0]) / self.time_step))
        upper = int(np.round((self.figure_options['x_limit_upper'].get() - self.time_vector[0]) / self.time_step))
        if start < lower or end > upper:
            messagebox.showerror('Error', 'Event out of bounds (Options -> X-Limit)')
            return

        # Flash the event
        def remove():
            if line[0] in ax.get_lines():
                line[0].remove()
                self.widgets['analysis_view']['fig'].canvas.draw()

        ax = self.widgets['analysis_view']['fig'].axes[0]
        trace_min = np.min(self.trace[start:end])
        trace_max = np.max(self.trace[start:end])
        difference = 0.5 * abs(trace_max - trace_min)
        ax.set(xlim=(start * self.time_step + self.time_vector[0], end * self.time_step + self.time_vector[0]),
               ylim=(trace_min - difference, trace_max + difference))
        line = ax.plot(self.time_vector[event_coordinates[0]:event_coordinates[1] + 1],
                       self.trace[event_coordinates[0]:event_coordinates[1] + 1], 'r', linewidth=3)
        self.widgets['analysis_view']['toolbar'].push_current()
        self.widgets['analysis_view']['fig'].canvas.draw()
        try:
            self.after(500, remove)
        except:
            pass

    def validate_x_limit(self, event, contents):
        try:
            if event == 'focusout' or event == '<Return>':
                # Empty on focus out is not allowed
                if contents == '':
                    self.set_x_limits()
                # Check if the order is correct
                upper = float(self.figure_options['x_limit_upper'].get())
                lower = float(self.figure_options['x_limit_lower'].get())
                if upper <= lower:
                    self.set_x_limits()
                # Check if the value is within bounds
                if self.time_vector is not None and (upper > self.time_vector[-1] or lower < self.time_vector[0]):
                    self.set_x_limits()
                self.update_figure(retain_view=False)
                return
            if event == 'key':
                # Allow the user to delete the contents
                if contents == '':
                    return True

                # Ensure the contents are a float (will raise exception if not)
                float(contents)
            return True
        except:
            if event == 'key':
                return False
            self.set_x_limits()
            return False

    def validate_digits(self, event, contents, name):
        try:
            if event == 'focusout' and contents == '':
                self.analysis_options[name].set(self.analysis_options[name].get())
            if event == 'key' and contents == '':
                return True

            float(contents)
            return True
        except:
            return False

    def set_x_limits(self):
        if self.time_vector is None:
            self.figure_options['x_limit_lower'].set(self.default_figure_options['x_limit_lower'])
            self.figure_options['x_limit_upper'].set(self.default_figure_options['x_limit_upper'])
        else:
            self.figure_options['x_limit_lower'].set(self.time_vector[0])
            self.figure_options['x_limit_upper'].set(self.time_vector[-1])

    def flash_entry(self, entry):
        current = entry.cget('style')

        self.style.configure(f'flash.{current}', bordercolor=self.style.colors.primary,
                             relief='raised', foreground=self.style.colors.primary)
        for i in np.arange(0, 10, 2):
            self.after(i * 250, lambda: entry.configure(style=f'flash.{current}'))
            self.after((i + 1) * 250, lambda: entry.configure(style=current))

    def toolbar_key_press(self, figure_id, event):
        key_press_handler(event, self.widgets[figure_id]['canvas'], self.widgets[figure_id]['toolbar'])

    def check_cutoff(self, cutoff):
        if not self.flags['loaded']:
            return

        # Show the initial estimate
        self.baseline = nn.baseline_filter(
            self.trace,
            cutoff,
            1 / self.time_step
        )
        self.update_figure(title=f'Cutoff = {cutoff}')

    def check_filtered(self):
        if not self.flags['loaded']:
            return

        # Show the initial estimate
        self.filtered_trace = nn.bounds_filter(self.trace, self.analysis_options['bounds_filter'].get())

    def process_results(self, results):
        if isinstance(results, list):
            self.analysis_results['Iteration'] = dict(
                trace_stats=[res.trace_stats for res in results[:-1]],
                event_stats=[res.event_stats for res in results[:-1]])
            label = results[-1].label
            self.analysis_results[label] = results[-1]
        else:
            label = results.label
            self.analysis_results[label] = results

        self.baseline = self.analysis_results[label].baseline
        self.threshold = self.analysis_results[label].threshold
        self.calculation_trace = self.analysis_results[label].calculation_trace
        self.event_coordinates = self.analysis_results[label].event_coordinates

        # Extract features
        events = Events(label)
        events.events = nn.simple_extractor(self.trace, self.baseline, self.event_coordinates, int(1 / self.time_step))
        # Store maximum cutoff
        max_cutoffs = nn.calculate_cutoffs(self.trace, self.baseline, self.threshold, self.event_coordinates,
                                           self.analysis_options['estimate_cutoff'].get(), int(1 / self.time_step))
        events.add_feature('Max Cutoff', max_cutoffs)
        self.analysis_results[label].events = events
        # Update the cutoff estimate window
        if label == 'Estimate':
            self.analysis_options['event_direction'].set(self.analysis_results[label].event_direction)
            self.flash_entry(self.widgets['analyse']['event_direction_input'])
            self.analysis_options['cutoff'].set(np.round(self.analysis_results[label].event_stats['Max Cutoff'], 2))
            self.flash_entry(self.widgets['analyse']['cutoff_input'])
        else:
            self.analysis_results[label].event_stats['Max Cutoff'] = np.percentile(max_cutoffs, 25)

        # Update the results window
        self.update_results()
        self.update()
        # Show the vectors
        self.update_figure(title=label)

    def load_trace(self):
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        path = tk.filedialog.askopenfilename(
            initialdir="/" if self.path is None else os.path.split(self.path)[0],
            title="Choose a file",
            filetypes=(("All", "*.abf *.csv *.tsv *.txt *.dat"), ("ABF files", "*.abf"), ("CSV files", "*.csv"),
                       ("TSV files", "*.tsv"), ("Text files", "*.txt"), ("Data files", "*.dat"))
        )
        if path == '' or path is None:
            return

        current_time = time.time()
        try:
            # Axon binary format
            if path.endswith('.abf'):
                data = pyabf.ABF(path)
                trace = data.sweepY.astype(float)
                time_vector = data.sweepX.astype(float)
            # Separated format
            elif path.endswith('.csv') or path.endswith('.tsv') or path.endswith('.txt') or path.endswith('.dat'):
                data = pd.read_csv(path, sep=None, engine='python')
                trace = np.array(data[data.keys()[1]], dtype=float)
                time_vector = np.array(data[data.keys()[0]], dtype=float)
            else:
                raise ValueError('File type not supported')
            time_step = time_vector[1] - time_vector[0]
        except Exception as e:
            messagebox.showerror('Error', f'Loading failed: {repr(e)}')
            return
        print(f'Loading time: {time.time() - current_time:.4f}')

        self.reset_all()
        # Update the trace vectors
        self.path = path
        self.trace = trace
        self.time_vector = time_vector
        self.time_step = time_step
        # Set the z-score
        self.analysis_options['z_score'].set(np.round(norm.ppf(1 - ((1 / len(self.trace)) / 2)), 3))
        self.default_analysis_options['z_score'] = self.analysis_options['z_score'].get()
        self.flash_entry(self.widgets['analyse']['z_score_input'])
        # Show the trace information and trace vector
        self.update_load()
        # Set the figure limits
        self.set_x_limits()
        self.update_figure(title='Trace', retain_view=False)

        # Set the flag
        self.flags['loaded'] = True

    def initial_estimate(self):
        if not self.flags['loaded']:
            messagebox.showerror('Error', 'Trace not loaded')
            return
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        current_time = time.time()
        self.widgets['analysis_view']['progress_bar'].start(10)

        # Filter the trace
        self.filtered_trace = nn.bounds_filter(self.trace, self.analysis_options['bounds_filter'].get())

        # Estimate the cutoff and direction
        def worker(results, errors,  *args):
            try:
                results.put(nn.initial_estimate(*args)[2])
            except Exception as e:
                errors.put(traceback.format_exc())
                errors.put(e)
                return

        # Run on a separate thread
        results = queue.Queue()
        errors = queue.Queue()
        thread = Thread(
            target=worker,
            args=(
                results,
                errors,
                self.trace,
                self.filtered_trace,
                int(1 / self.time_step),
                self.analysis_options['estimate_cutoff'].get(),
                self.analysis_options['threshold_window'].get(),
                self.analysis_options['z_score'].get(),
            )
        )
        thread.start()
        self.flags['running'] = True
        while thread.is_alive():
            self.update()
            if not errors.empty():
                self.widgets['analysis_view']['progress_bar'].stop()
                self.flags['running'] = False
                messagebox.showerror('Error', f'Estimate failed: {repr(errors.get())}')
                return
        thread.join()
        self.update_idletasks()
        self.flags['running'] = False
        results = results.get()

        self.widgets['analysis_view']['progress_bar'].stop()
        print(f'Estimate time: {time.time() - current_time:.4f}')

        # Update analysis results with estimate results
        self.process_results(results)
        # Set the focus
        self.widgets['analyse']['iterate'].focus_set()
        # Set the flag
        self.flags['estimated'] = True

    def iterate(self):
        if not self.flags['loaded']:
            messagebox.showerror('Error', 'Trace not loaded')
            return
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        current_time = time.time()
        self.widgets['analysis_view']['progress_bar'].start(10)

        # Filter the trace
        self.filtered_trace = nn.bounds_filter(self.trace, self.analysis_options['bounds_filter'].get())

        # Iterate to find the final baseline and threshold
        def worker(results, errors, *args):
            try:
                if self.analysis_options['parallel'].get():
                    results.put(nn.parallel_iterate(*args)[2])
                else:
                    results.put(nn.iterate(*args)[2])
            except Exception as e:
                errors.put(traceback.format_exc())
                errors.put(e)
                return

        # Run on a separate thread
        results = queue.Queue()
        errors = queue.Queue()
        thread = Thread(
            target=worker,
            args=(
                results,
                errors,
                self.trace,
                self.filtered_trace,
                int(1 / self.time_step),
                self.analysis_options['cutoff'].get(),
                self.analysis_options['event_direction'].get(),
                self.analysis_options['replace_factor'].get(),
                self.analysis_options['replace_gap'].get(),
                self.analysis_options['threshold_window'].get(),
                self.analysis_options['z_score'].get()
            )
        )
        thread.start()
        self.flags['running'] = True
        while thread.is_alive():
            self.update()
            if not errors.empty():
                self.widgets['analysis_view']['progress_bar'].stop()
                self.flags['running'] = False
                messagebox.showerror('Error', f'Iteration failed: {repr(errors.get())}')
                return
        thread.join()
        self.update_idletasks()
        self.flags['running'] = False
        results = results.get()

        self.widgets['analysis_view']['progress_bar'].stop()
        print(f'Iteration time: {time.time() - current_time:.4f}')

        # Update analysis results with iteration results
        self.process_results(results)
        # Set the focus
        self.widgets['results']['save'].focus_set()
        # Set the flag
        self.flags['iterated'] = True

    def save_results(self):
        if not self.flags['iterated'] and not self.flags['estimated']:
            messagebox.showerror('Error', 'No results to save')
            return
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return
        if self.flags['iterated']:
            results = self.analysis_results['Final'].events
            label = 'final'
        else:
            results = self.analysis_results['Estimate'].events
            label = 'estimate'
        if not results:
            messagebox.showerror('Error', 'No results to save')
            return

        # Save the results
        path = tk.filedialog.asksaveasfile(
            initialfile=basename(self.path).split('.')[0] + f'_{label}.csv',
            defaultextension=".csv",
            initialdir=os.path.split(self.path)[0],
            title="Choose a location",
            filetypes=([("CSV files", "*.csv")])
        )
        if path == '' or path is None:
            return

        # If you don't do this, DictWriter will ruin your life :(
        for event in results:
            event['Vector'] = list(event['Vector'])

        with open(path.name, 'w', encoding='utf8', newline='') as path:
            w = csv.DictWriter(path, results.events[0].keys())
            w.writeheader()
            w.writerows(results.events)

        # Success
        messagebox.showinfo('Success', f'Saved {label} results')

        # Set the flag
        self.flags['saved'] = True
