# cython: infer_types=True

import csv
import traceback
import cython
import os.path
import queue
import time
import sys
import copy
import base64
import struct
import tkinter as tk
from json import loads, dumps, JSONDecodeError
from functools import partial
from webbrowser import open_new
from idlelib import tooltip
from os.path import basename
from threading import Thread
from tkinter import ttk, colorchooser, messagebox, filedialog
from tkinter.font import Font
import numpy as np
import pyabf
from ast import literal_eval
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib.widgets import RangeSlider
from matplotlib import patheffects as pe, rc, style as mplstyle
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from ttkbootstrap import Style, Floodgauge
from notnormal import not_normal as nn
from notnormal.results import Events

COMPILED = cython.compiled
PADDING = 6
WINDOW_PADDING = 3
ENTRY_WIDTH = 9
LARGE_FONT = 16
MEDIUM_FONT = 14
SMALL_FONT = 10

class CustomFigureCanvas(FigureCanvasTkAgg):
    def __init__(self, figure, window, root):
        super().__init__(figure, window)
        self.root = root
        self.after_id = None

    def resize(self, event):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        func = super(CustomFigureCanvas, self).resize
        self.after_id = self.root.after(50, func, event)

class FeatureWindow(tk.Toplevel):
    def __init__(self, master, events):
        super().__init__(master)
        self.master = master
        self.events = events
        # Loose focus pls
        self.attributes('-topmost', 1)
        # Title
        self.title("Feature Window")
        # Background
        self.configure(bg="black")
        # Icon
        if sys.platform == 'win32':
            self.iconbitmap(True, master.icon)
        else:
            self.iconphoto(True, master.icon)
        # Root window size and resizability
        self.geometry(f'{master.width // 2}x{master.height // 2}+{master.width // 4}+{master.height // 4}')
        self.state('normal')
        self.resizable(True, True)

        # Dictionary to store all widgets
        self.widgets = dict()
        # Dictionary to store all windows
        self.windows = dict()
        # Figure options
        self.plot_options = dict()
        # Axis options
        self.axis_options = dict()
        # Axis functions
        self.axis_func_options = dict()
        # Feature filters
        self.feature_filters = dict()
        # Features
        self.features = list(events[1].keys())
        for feature in list(events[1].keys()):
            if type(events[1][feature]) in [np.ndarray, list, tuple]:
                self.features.remove(feature)
        # Plot types
        self.figure_configs = []
        self.current_figure = None
        figure_configs = [
            {
                "label": "Histogram",
                "function": "hist",
                "sub_plot_init": {"projection": None},
                "axis_init": {},
                "axis_func_init": {
                    "grid": {"visible": False},
                },
                "plot_init": {"zorder": 2},
                "axis_options": {},
                "axis_func_options": {
                    "grid": {"label": "Grid", "relim": False, "type": "bool", "default": True, "kwargs": {}},
                },
                "plot_options": {
                    "x": {"label": "Feature", "relim": True, "type": "feature", "values": self.features, "default": self.features[1]},
                    "bins": {"label": "Bins", "relim": True, "type": "int", "values": [1, 100, 1], "default": 20},
                    "density": {"label": "Density", "relim": True, "type": "bool", "default": False},
                    "cumulative": {"label": "Cumulative", "relim": True, "type": "bool", "default": False},
                    "rwidth": {"label": "Relative Width", "relim": False, "type": "float", "values": [0, 1, 0.05], "default": 0.8},
                    "facecolor": {"label": "Face Colour", "relim": False, "type": "colour", "default": master.colours["trace"]},
                    "edgecolor": {"label": "Edge Colour", "relim": False, "type": "colour", "default": master.colours["baseline"]}
                },
                "feature_filters": {
                    "Outlier": {"label": "Show Outliers", "relim": False, "type": "bool", "onvalue": "all", "default": False},
                    "Direction": {"label": "Direction", "relim": False, "type": "str", "values": ["up", "down", "all"], "default": "all"},
                }
            },
            {
                "label": "2D Scatter",
                "function": "scatter",
                "sub_plot_init": {"projection": None},
                "axis_init": {},
                "axis_func_init": {
                    "grid": {"visible": True},
                },
                "plot_init": {"zorder": 2},
                "axis_options": {},
                "axis_func_options": {
                    "grid": {"label": "Grid", "relim": False, "type": "bool", "default": True, "kwargs": {}},
                },
                "plot_options": {
                    "x": {"label": "Feature 1", "relim": True, "type": "feature", "values": self.features, "default": self.features[1]},
                    "y": {"label": "Feature 2", "relim": True, "type": "feature", "values": self.features, "default": self.features[2]},
                    "s": {"label": "Marker Size", "relim": False, "type": "int", "values": [1, 100, 1], "default": 40},
                    "linewidths": {"label": "Marker Outline", "relim": False, "type": "float", "values": [0, 5, 0.5], "default": 1},
                    "marker": {"label": "Marker Type", "relim": False, "type": "str", "values": ["o", ".", "*", "s", "p"], "default": "o"},
                    "alpha": {"label": "Alpha", "relim": False, "type": "float", "values": [0, 1, 0.05], "default": 1},
                    "c": {"label": "Face Colour", "relim": False, "type": "colour", "default": master.colours["trace"]},
                    "edgecolor": {"label": "Edge Colour", "relim": False, "type": "colour", "default": master.colours["baseline"]}
                },
                "feature_filters": {
                    "Outlier": {"label": "Show Outliers", "relim": False, "type": "bool", "onvalue": "all", "default": False},
                    "Direction": {"label": "Direction", "relim": False, "type": "str", "values": ["up", "down", "all"], "default": "all"},
                }
            },
            {
                "label": "3D Scatter",
                "function": "scatter",
                "sub_plot_init": {"projection": "3d"},
                "axis_init": {},
                "axis_func_init": {
                    "grid": {"visible": True},
                },
                "plot_init": {"zorder": 2},
                "axis_options": {},
                "axis_func_options": {},
                "plot_options": {
                    "xs": {"label": "Feature 1", "relim": True, "type": "feature", "values": self.features, "default": self.features[1]},
                    "ys": {"label": "Feature 1", "relim": True, "type": "feature", "values": self.features, "default": self.features[2]},
                    "zs": {"label": "Feature 3", "relim": True, "type": "feature", "values": self.features, "default": self.features[3]},
                    "s": {"label": "Marker Size", "relim": False, "type": "int", "values": [1, 100, 1], "default": 40},
                    "linewidths": {"label": "Marker Outline", "relim": False, "type": "float", "values": [0, 5, 0.5], "default": 1},
                    "marker": {"label": "Marker Type", "relim": False, "type": "str", "values": ["o", ".", "*", "s", "p"], "default": "o"},
                    "alpha": {"label": "Alpha", "relim": False, "type": "float", "values": [0, 1, 0.05], "default": 1},
                    "c": {"label": "Face Colour", "relim": False, "type": "colour", "default": master.colours["trace"]},
                    "edgecolor": {"label": "Edge Colour", "relim": False, "type": "colour", "default": master.colours["baseline"]}
                },
                "feature_filters": {
                    "Outlier": {"label": "Show Outliers", "relim": False, "type": "bool", "onvalue": "all", "default": False},
                    "Direction": {"label": "Direction", "relim": False, "type": "str", "values": ["up", "down", "all"], "default": "all"},
                }
            }
        ]

        # Create layout
        self.init_layout()
        # Create options
        self.init_options()
        # Create feature view
        self.init_feature_view()
        # Create plots
        for config in figure_configs:
            self.add_figure_config(config)

        # Select the first tab
        self.set_current_tab(self.figure_configs[0]['label'])

    def init_layout(self):
        # Main window
        self.windows['main'] = ttk.Frame(self)
        self.windows['main'].pack(fill=tk.BOTH, expand=True)
        # Left window
        self.windows['left'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')
        # Right window
        self.windows['right'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')

        # Main layout
        self.windows['main'].rowconfigure(0, weight=1)
        self.windows['main'].columnconfigure(1, weight=1)
        self.windows['left'].grid(row=0, column=0, sticky="nsew")
        self.windows['right'].grid(row=0, column=1, sticky="nsew")

        # Left (options window)
        self.windows['options'] = ttk.Frame(self.windows['left'], style='primary.TFrame')
        # Right (feature view window)
        self.windows['feature_view'] = ttk.Frame(self.windows['right'], style='primary.TFrame')

        # Left layout
        self.windows['left'].columnconfigure(0, weight=1, minsize=self.master.width // 6)
        self.windows['left'].rowconfigure(0, weight=1)
        self.windows['options'].grid(row=0, column=0, sticky="nsew", padx=(WINDOW_PADDING, 0), pady=WINDOW_PADDING)
        # Right layout
        self.windows['right'].rowconfigure(0, weight=1)
        self.windows['right'].columnconfigure(0, weight=1)
        self.windows['feature_view'].grid(row=0, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

    def init_options(self):
        self.widgets['options'] = dict()
        # Options title
        self.widgets['options']['title'] = ttk.Label(self.windows['options'], text="Options",
                                                     style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['options_internal'] = ttk.Frame(self.windows['options'])
        # Layout
        self.windows['options'].columnconfigure(0, weight=1)
        self.windows['options'].rowconfigure(1, weight=1)
        self.widgets['options']['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        self.windows['options_internal'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

        # Options notebook
        self.widgets['options']['notebook'] = ttk.Notebook(self.windows['options_internal'])
        self.widgets['options']['notebook'].enable_traversal()
        self.widgets['options']['notebook'].bind("<<NotebookTabChanged>>", lambda _: self.set_current_figure())
        # Layout
        self.windows['options_internal'].columnconfigure(0, weight=1)
        self.windows['options_internal'].rowconfigure(0, weight=1)
        self.widgets['options']['notebook'].grid(row=0, column=0, sticky="nsew")

    def init_feature_view(self):
        self.widgets['feature_view'] = dict()
        # Figure title
        self.widgets['feature_view']['title'] = ttk.Label(self.windows['feature_view'], text="Feature View",
                                                          style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['feature_view_internal'] = ttk.Frame(self.windows['feature_view'])
        # Layout
        self.windows['feature_view'].columnconfigure(0, weight=1)
        self.windows['feature_view'].rowconfigure(1, weight=1)
        self.widgets['feature_view']['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        self.windows['feature_view_internal'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

    def create_options_tab(self, figure_config: dict, index: int = None):
        label = figure_config['label']

        # Tab widgets
        self.widgets['options'][label] = dict()
        widgets = self.widgets['options'][label]

        # Tab frame and separator
        widgets['frame'] = ttk.Frame(self.widgets['options']['notebook'])
        widgets['header_separator'] = ttk.Separator(widgets['frame'], style='primary.TSeparator')
        # Tab layout (column 0 is for labels, column 1 is for inputs)
        widgets['frame'].columnconfigure(0, weight=1, uniform=label)
        widgets['frame'].columnconfigure(1, weight=2, uniform=label)
        widgets['header_separator'].grid(row=0, column=0, columnspan=2, sticky="nsew")
        # Add tab
        if index is None or index >= self.widgets['options']['notebook'].index('end'):
            self.widgets['options']['notebook'].add(widgets['frame'], text=label)
        else:
            self.widgets['options']['notebook'].insert(index, widgets['frame'], text=label)

        # Plot and axis options
        self.plot_options[label] = dict()
        self.axis_options[label] = dict()
        self.axis_func_options[label] = dict()
        self.feature_filters[label] = dict()
        opt_length = (len(figure_config['axis_options']) + len(figure_config['plot_options']) +
                      len(figure_config['axis_func_options']) + len(figure_config['feature_filters']))

        i = 1
        for tk_var, options in [(self.plot_options[label], figure_config['plot_options']),
                                (self.axis_options[label], figure_config['axis_options']),
                                (self.axis_func_options[label], figure_config['axis_func_options']),
                                (self.feature_filters[label], figure_config['feature_filters'])]:
            for opt in options.keys():
                # Tk variables
                if options[opt]['type'] in ['str', 'colour', 'feature']:
                    tk_var[opt] = tk.StringVar()
                elif options[opt]['type'] == 'int':
                    tk_var[opt] = tk.IntVar()
                elif options[opt]['type'] == 'float':
                    tk_var[opt] = tk.DoubleVar()
                elif options[opt]['type'] == 'bool':
                    tk_var[opt] = tk.BooleanVar()

                # Default
                tk_var[opt].set(options[opt]['default'])

                # Label
                widgets['frame'].rowconfigure(i, weight=1, uniform=label)
                widgets[opt + '_label'] = ttk.Label(widgets['frame'], text=options[opt]['label'])
                widgets[opt + '_label'].grid(row=i, column=0, sticky="nsew", padx=PADDING)

                # Entry
                if options[opt]['type'] in ['str', 'feature']:
                    widgets[opt] = ttk.Combobox(
                        widgets['frame'],
                        textvariable=tk_var[opt],
                        width=ENTRY_WIDTH,
                        justify='center',
                        values=options[opt]['values'],
                        state='readonly'
                    )
                    widgets[opt].bind("<<ComboboxSelected>>", lambda _, x = options[opt]['relim']: (
                                                                self.update_figure(figure_config, relim=x)))
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=PADDING)
                    widgets[opt].configure(font=self.master.fonts['small'])
                elif options[opt]['type'] in ['int', 'float']:
                    widgets[opt] = ttk.Spinbox(
                        widgets['frame'],
                        textvariable=tk_var[opt],
                        width=ENTRY_WIDTH,
                        justify='center',
                        from_=options[opt]['values'][0],
                        to=options[opt]['values'][1],
                        increment=options[opt]['values'][2],
                        command=partial(self.update_figure, figure_config, relim=options[opt]['relim'])
                    )
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=PADDING)
                    widgets[opt].configure(font=self.master.fonts['small'])
                elif options[opt]['type'] == 'bool':
                    widgets[opt] = ttk.Checkbutton(
                        widgets['frame'],
                        variable=tk_var[opt],
                        width=ENTRY_WIDTH - 5,
                        onvalue=True,
                        offvalue=False,
                        command=partial(self.update_figure, figure_config, relim=options[opt]['relim']),
                        style='Roundtoggle.Toolbutton'
                    )
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=PADDING)
                elif options[opt]['type'] == 'colour':
                    widgets[opt] = ttk.Button(
                        widgets['frame'],
                        command=partial(self.update_colour, figure_config, opt),
                        width=ENTRY_WIDTH - 3
                    )
                    self.master.style.configure(
                        f'{label.replace(" ", "")}_{opt}_colour.TButton', background=tk_var[opt].get(), borderwidth=0)
                    self.master.style.map(f'{label.replace(" ", "")}_{opt}_colour.TButton',
                                          background=[('active', tk_var[opt].get())])
                    widgets[opt].configure(style=f'{label.replace(" ", "")}_{opt}_colour.TButton')
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=PADDING, pady=6)

                # Separator
                if i < (2 * opt_length) - 1:
                    widgets[opt + '_separator'] = ttk.Separator(widgets['frame'], orient='horizontal')
                    widgets[opt + '_separator'].grid(row=i + 1, column=0, columnspan=2, sticky="nsew")

                i += 2

        # Add config button
        widgets['config_button'] = ttk.Button(widgets['frame'], text="Edit Configuration", underline=0,
                                              command=partial(self.create_edit_window, figure_config),
                                              style='secondary.Outline.TButton')
        self.bind('<Control-a>', lambda _: partial(self.create_edit_window, figure_config))
        widgets['config_button'].grid(row=i, column=0, columnspan=2, sticky="nsew", pady=(0, PADDING), padx=PADDING)

    def create_figure_frame(self, figure_config: dict):
        label = figure_config['label']
        sub_plot_init = figure_config['sub_plot_init']

        # Figure frame widgets
        self.widgets['feature_view'][label] = dict()
        widgets = self.widgets['feature_view'][label]


        widgets['frame'] = ttk.Frame(self.windows['feature_view_internal'])
        # configure layout (do not place frame in grid as this is done dynamically)
        self.windows['feature_view_internal'].columnconfigure(0, weight=1)
        self.windows['feature_view_internal'].rowconfigure(0, weight=1)
        # Figure and canvas
        widgets['fig'] = Figure(layout='constrained', figsize=(1, 1))
        widgets['canvas'] = FigureCanvasTkAgg(widgets['fig'], widgets['frame'])
        widgets['canvas'].draw()
        # Toolbar
        widgets['toolbar'] = NavigationToolbar2Tk(widgets['canvas'], widgets['frame'], pack_toolbar=False)
        for child in widgets['toolbar'].winfo_children():
            child.pack_forget()
            child.configure(background='white', border=1)
        for child in widgets['toolbar'].winfo_children():
            if child != widgets['toolbar'].children['!button4']:
                child.pack(side=tk.LEFT)
        widgets['toolbar'].update()
        # Toolbar separator
        widgets['toolbar_separator'] = ttk.Separator(widgets['frame'])
        # Frame layout
        widgets['frame'].columnconfigure(0, weight=1)
        widgets['frame'].rowconfigure(2, weight=1)
        widgets['toolbar'].grid(row=0, column=0, sticky='nsew', padx=PADDING)
        widgets['toolbar_separator'].grid(row=1, column=0, sticky="nsew")
        widgets['canvas'].get_tk_widget().grid(row=2, column=0, sticky='nsew', padx=2 * PADDING, pady=2 * PADDING)

        # Feature view
        widgets['fig'].add_subplot(111, **sub_plot_init)

        # Bind events
        widgets['canvas'].mpl_connect("pick_event", self.pick_event)

    def update_colour(self, figure_config: dict, option: str):
        label = figure_config['label']
        plot_options = self.plot_options[label]
        widgets = self.widgets['options'][label]

        colour = colorchooser.askcolor(title="Choose color", initialcolor=plot_options[option].get())[1]
        if colour is None:
            return

        plot_options[option].set(colour)
        self.master.style.configure(
            f'{label.replace(" ", "")}_{option}_colour.TButton', background=plot_options[option].get(),)
        self.master.style.map(f'{label.replace(" ", "")}_{option}_colour.TButton',
                              background=[('active', plot_options[option].get())])
        widgets[option].configure(style=f'{label.replace(" ", "")}_{option}_colour.TButton')

        self.update_figure(figure_config)

    def update_figure(self, figure_config: dict, relim: bool = False):
        label = figure_config['label']
        function = figure_config['function']
        plot_init = figure_config['plot_init']
        plot_options = figure_config['plot_options']
        axis_init = figure_config['axis_init']
        axis_options = figure_config['axis_options']
        axis_func_init = figure_config['axis_func_init']
        axis_func_options = figure_config['axis_func_options']

        # Get and clear axis
        ax = self.widgets['feature_view'][label]['fig'].axes[0]
        view_limits = (ax.get_xlim(), ax.get_ylim())
        ax.clear()

        # Get the filtered event IDs
        event_ids = self.get_filtered_ids(figure_config)

        # Get figure options
        options = dict()
        current_features = []
        for option in plot_options.keys():
            if plot_options[option]['type'] == 'feature':
                options[option] = self.events.get_feature(self.plot_options[label][option].get(), event_ids)
                current_features.append(self.plot_options[label][option].get())
            else:
                options[option] = self.plot_options[label][option].get()

        # Plot
        plot = getattr(ax, function)(**options, **plot_init, picker=True)

        # Get axis options
        options = dict()
        for option in axis_options.keys():
            options[option] = self.axis_options[label][option].get()

        # Set axis options
        ax.set(**options, **axis_init, label=str(current_features))

        # Title
        if 'title' not in axis_init.keys():
            ax.set_title(self.events.label, fontsize=LARGE_FONT)
        # Labels
        if 'xlabel' not in axis_init.keys():
            ax.set_xlabel(current_features[0], fontsize=MEDIUM_FONT)
        if 'ylabel' not in axis_init.keys() and len(current_features) > 1:
            ax.set_ylabel(current_features[1], fontsize=MEDIUM_FONT)
        if 'zlabel' not in axis_init.keys() and len(current_features) > 2:
            ax.set_zlabel(current_features[2], rotation=90, fontsize=MEDIUM_FONT)

        # Set default axis func options
        for option in axis_func_init.keys():
            getattr(ax, option)(**axis_func_init[option])

        # Set axis func options
        for option in axis_func_options.keys():
            getattr(ax, option)(self.axis_func_options[label][option].get(),
                                **axis_func_options[option].get('kwargs', {}))


        # Update data limits
        if relim:
            try:
                ax.ignore_existing_data_limits = True
                ax.update_datalim(plot.get_datalim(ax.transData))
            except AttributeError:
                ax.relim()
            ax.autoscale(True)
            self.widgets['feature_view'][label]['toolbar'].update()
            self.widgets['feature_view'][label]['toolbar'].push_current()
        else:
            ax.set_xlim(view_limits[0])
            ax.set_ylim(view_limits[1])

        # Draw
        self.widgets['feature_view'][label]['canvas'].draw_idle()
        self.widgets['feature_view'][label]['canvas'].flush_events()

    def update_current_figure(self, relim: bool = False):
        config = self.figure_configs[self.current_figure]
        self.update_figure(config, relim)

    def update_all_figures(self, relim: bool = False):
        for config in self.figure_configs:
            self.update_figure(config, relim)

    def create_edit_window(self, figure_config: dict):
        self.windows['edit_config'] = dict()
        self.windows['edit_config']['windows'] = dict()
        self.windows['edit_config']['widgets'] = dict()
        windows = self.windows['edit_config']['windows']
        widgets = self.windows['edit_config']['widgets']

        windows['top_level'] = tk.Toplevel(self)
        windows['top_level'].attributes('-topmost', 1)
        windows['top_level'].title("Edit Configuration")
        windows['top_level'].configure(bg="black")
        if sys.platform == 'win32':
            windows['top_level'].iconbitmap(True, self.master.icon)
        else:
            windows['top_level'].iconphoto(True, self.master.icon)
        windows['top_level'].geometry(
            f'{self.master.width // 2}x{self.master.height // 2}+{self.master.width // 4}+{self.master.height // 4}')
        windows['top_level'].state('normal')
        windows['top_level'].resizable(True, True)

        # Main window
        windows['main'] = ttk.Frame(windows['top_level'])
        windows['main'].pack(fill=tk.BOTH, expand=True)
        # Center window
        windows['center'] = ttk.Frame(windows['main'], style='secondary.TFrame')

        # Main layout
        windows['main'].rowconfigure(0, weight=1)
        windows['main'].columnconfigure(0, weight=1)
        windows['center'].grid(row=0, column=0, sticky="nsew")

        # Options window
        windows['config'] = ttk.Frame(windows['center'], style='primary.TFrame')

        # Layout
        windows['center'].columnconfigure(0, weight=1)
        windows['center'].rowconfigure(0, weight=1)
        windows['config'].grid(row=0, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

        # Title and inner frame
        widgets['title'] = ttk.Label(windows['config'], text="JSON Configuration", style='primary.Inverse.TLabel',
                                     anchor='center')
        windows['config_internal'] = ttk.Frame(windows['config'])
        # Layout
        windows['config'].columnconfigure(0, weight=1)
        windows['config'].rowconfigure(1, weight=1)
        widgets['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        windows['config_internal'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

        # Entry
        size = self.master.fonts['small'].measure('    ')
        widgets['entry'] = tk.Text(windows['config_internal'], tabs=size, font=self.master.fonts['small'])
        windows['config_internal'].columnconfigure(0, weight=1, uniform=1)
        windows['config_internal'].rowconfigure(0, weight=1, uniform=1)
        widgets['entry'].grid(row=0, column=0, sticky="nsew", padx=PADDING, pady=(0, PADDING))

        widgets['add_button'] = ttk.Button(windows['config_internal'], text="Confirm", underline=0,
                                           command=partial(self.edit_figure_config, figure_config),
                                           style='secondary.Outline.TButton')
        widgets['add_button'].grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, PADDING), padx=PADDING)

        # Create a new tag
        widgets['entry'].tag_configure('JSON', foreground=self.master.colours['trace'])

        # Convert to json
        widgets['entry'].insert(tk.END, dumps(figure_config, indent=4))

    def edit_figure_config(self, figure_config: dict):
        # Get the new config
        config = self.windows['edit_config']['widgets']['entry'].get('1.0', 'end')

        # Replace 'features' with the features
        config = config.replace('features', str(self.features))
        for i in range(100):
            if f'features[{i}]' in config:
                config = config.replace(f'features[{i}]', f'"{self.features[i]}"')

        # Convert the config back to a dictionary
        try:
            config = loads(config)
        except JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON configuration")
            return

        # Parse the config
        try:
            config = self.parse_figure_config(config)
        except ValueError as e:
            messagebox.showerror("Error", e)
            return

        # Replace the old config
        current_config = self.get_figure_config(figure_config['label'])
        current_config.update(config)
        # Get the tab index
        index = self.widgets['options']['notebook'].index(self.widgets['options'][figure_config['label']]['frame'])
        # Remove the old figure and tab
        self.remove_figure_config(current_config)

        # Add figure frame
        self.create_figure_frame(figure_config)
        # Add options tab
        self.create_options_tab(figure_config, index)
        self.set_current_tab(figure_config['label'])
        # Update the figure
        self.update_figure(figure_config, relim=True)

        # Destroy the window
        self.windows['edit_config']['windows']['top_level'].destroy()

    def add_figure_config(self, figure_config: dict):
        # Parse the figure configuration
        if figure_config['label'] in [config['label'] for config in self.figure_configs]:
            messagebox.showerror("Error", "label already exists in figure configurations")
            return
        try:
            figure_config = self.parse_figure_config(figure_config)
        except ValueError as e:
            messagebox.showerror("Error", e)
            return

        # Add to the list of figures and configs
        self.figure_configs.append(figure_config)
        # Add figure frame
        self.create_figure_frame(figure_config)
        # Add options tab
        self.create_options_tab(figure_config)
        # Update the figure
        self.update_figure(figure_config, relim=True)

    def remove_figure_config(self, figure_config: dict):
        # Remove the figure frame
        self.widgets['feature_view'][figure_config['label']]['frame'].destroy()
        # Remove the options tab
        self.widgets['options'][figure_config['label']]['frame'].destroy()
        # Update the current figure
        self.current_figure = None
        self.set_current_figure()

    @staticmethod
    def parse_figure_config(figure_config: dict):
        # Label
        if not 'label' in figure_config.keys():
            raise ValueError("label not found in figure configuration")
        if type(figure_config['label']) != str:
            raise ValueError("label must be a string")

        # Function
        if not 'function' in figure_config.keys():
            raise ValueError("function not found in figure configuration")
        if type(figure_config['function']) != str:
            raise ValueError("function must be a string")
        if not hasattr(Axes, figure_config['function']):
            raise ValueError("function not found in matplotlib Axes")

        # Optional config keys
        for key in ['sub_plot_init', 'axis_init', 'axis_func_init', 'plot_init', 'axis_options', 'axis_func_options',
                    'plot_options', 'feature_filters']:
            if not key in figure_config.keys() or figure_config[key] is None:
                figure_config[key] = {}
            if type(figure_config[key]) != dict:
                raise ValueError(f"{key} must be a dictionary")

        # Options keys
        for key in ['axis_options', 'axis_func_options', 'plot_options', 'feature_filters']:
            for subkey in figure_config[key].keys():
                if type(figure_config[key][subkey]) != dict:
                    raise ValueError(f"{subkey} in {key} must be a dictionary")
                if not 'label' in figure_config[key][subkey].keys():
                    raise ValueError("{subkey} in {key} must have a label")
                if not 'relim' in figure_config[key][subkey].keys():
                    raise ValueError("{subkey} in {key} must have a relim option")
                if not 'type' in figure_config[key][subkey].keys():
                    raise ValueError("{subkey} in {key} must have a type option")
                if not 'default' in figure_config[key][subkey].keys():
                    raise ValueError("{subkey} in {key} must have a default option")

        # Plot options
        if not len(figure_config['plot_options']):
            raise ValueError("plot_options cannot be empty")

        # Axis functions
        for key in ['axis_func_init', 'axis_func_options']:
            for subkey in figure_config[key].keys():
                if not hasattr(Axes, subkey):
                    raise ValueError(f"{subkey} not found in matplotlib Axes")

        return figure_config

    def get_filtered_ids(self, figure_config: dict):
        label = figure_config['label']
        feature_filters = figure_config['feature_filters']

        # Get feature filters
        filters = [(key, self.feature_filters[label][key].get()) for key in feature_filters.keys()]
        # Remove 'all' values
        items = []
        for key, value in filters:
            if value == 'all':
                continue
            if value == True and feature_filters[key].get('onvalue') == 'all':
                continue
            if value == False and feature_filters[key].get('offvalue') == 'all':
                continue
            items.append((key, value))

        event_ids = []
        # Get event IDs that match the filters
        for event in self.events:
            if not all([event[key] == value for key, value in items]):
                continue
            else:
                event_ids.append(event['ID'])

        return event_ids

    def get_figure_config(self, label: str):
        for config in self.figure_configs:
            if config['label'] == label:
                return config

    def set_current_tab(self, label: str):
        for i in self.widgets['options']['notebook'].tabs():
            if self.widgets['options']['notebook'].tab(i, "text") == label:
                self.widgets['options']['notebook'].select(i)
                break

    def set_current_figure(self):
        # Hide the previous plot
        if self.current_figure:
            self.widgets['feature_view'][self.current_figure]['frame'].grid_forget()
        # Currently selected tab
        tab_id = self.widgets['options']['notebook'].select()
        self.current_figure = self.widgets['options']['notebook'].tab(tab_id, "text")
        # Show the new plot
        self.widgets['feature_view'][self.current_figure]['frame'].grid(row=0, column=0, sticky="nsew")

    def pick_event(self, event):
        point = []
        if isinstance(event.artist, Line2D):
            point = [event.artist.get_xdata()[event.ind[0]], event.artist.get_ydata()[event.ind[0]]]
        elif isinstance(event.artist, Path3DCollection):
            point = [event.artist._offsets3d[0][event.ind[0]], event.artist._offsets3d[1][event.ind[0]],
                     event.artist._offsets3d[2][event.ind[0]]]
        elif isinstance(event.artist, PathCollection):
            point = [event.artist.get_offsets()[event.ind[0]][0], event.artist.get_offsets()[event.ind[0]][1]]
        elif isinstance(event.artist, Rectangle):
            point = [event.artist.get_x()]

        feature_names = literal_eval(event.canvas.figure.axes[0].get_label())

        if len(feature_names) != len(point):
            return

        event_id = None
        for e in self.events:
            if all([point[i] == e[feature_names[i]] for i in range(len(feature_names))]):
                event_id = e['ID']
                break

        if event_id is None:
            return

        # Get currently shown events
        label = self.events.label

        # Select the tab
        for i in self.master.widgets['results']['notebook'].tabs():
            if self.master.widgets['results']['notebook'].tab(i, "text") == label:
                self.master.widgets['results']['notebook'].select(i)
                break

        # Get the treeview
        tab = self.master.widgets['results']['notebook'].nametowidget(
            self.master.widgets['results']['notebook'].select())
        tree = tab.winfo_children()[2].winfo_children()[0]

        # Select the event
        for item in tree.get_children():
            if tree.item(item, 'text') == str(event_id):
                tree.selection_set(item)
                tree.focus(item)
                tree.see(item)
                dummy = tk.Event()
                dummy.widget = tree
                self.master.jump_to_event(label, dummy)
                break

class NotNormalGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Loose focus pls
        self.bind("<FocusOut>", lambda event: self.wm_attributes('-topmost', 0))
        # Title
        self.title("Not Normal")
        self.tk.call('tk', 'appname', 'NotNormal')
        # Background
        self.configure(bg="black")
        # Icon
        if hasattr(sys, '_MEIPASS'):
            self.data_path = os.path.join(sys._MEIPASS, 'data')
        else:
            self.data_path = os.path.join(__file__, '..', 'data')
        if sys.platform == 'win32':
            self.icon = os.path.join(self.data_path, 'logo.ico')
            self.iconbitmap(True, self.icon)
        else:
            self.icon = tk.PhotoImage(file=os.path.join(self.data_path, 'logo.png'))
            self.iconphoto(True, self.icon)
        # Root window size and resizability
        self.height = self.winfo_screenheight()
        self.width = self.winfo_screenwidth()
        self.geometry(f'{self.width}x{self.height}')
        self.state('zoomed')
        self.resizable(True, True)
        # DPI scaling
        self.tk.call('tk', 'scaling', self.winfo_fpixels('1i') / 72)
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
        # Temporary analysis options
        self.analysis_options['parallel'] = tk.BooleanVar(value=True)
        self.default_analysis_options['parallel'] = self.analysis_options['parallel'].get()
        # Analysis results
        self.analysis_results = dict()
        self.table_columns = ['ID', 'Area', 'Max Cutoff', 'Duration', 'Amplitude']
        # Trace information variables
        self.trace_information = dict()
        self.trace = None
        self.time_vector = None
        self.time_step = None
        self.filtered_trace = None
        self.baseline = None
        self.threshold = None
        self.initial_threshold = None
        self.calculation_trace = None
        self.event_coordinates = None
        self.current_cutoff = None
        # Flags for ordering
        self.flags = dict(loaded=False, estimated=False, iterated=False, saved=False, running=False, feature_win=False)
        self.after_id = None

        # Initialise the style
        self.style = Style(theme='pulse')
        self.fonts = None
        self.colours = None
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
        # Create options
        self.init_analysis_view_options()
        # Create results
        self.init_results()
        # Create tooltips
        self.init_tooltips()
        # Create and show the landing page
        self.init_landing_page()
        # Show the landing page window
        self.toggle_landing_page('show')

    def init_style(self):
        # Initialise the colours
        self.colours = {
            'baseline': '#FF0707',
            'threshold': '#FF9A52',
            'trace': '#5E3C99',
            'calculation_trace': '#B2ABD2',
            'filtered_trace': '#008837',
            'events': '#2dbd86'
        }
        # Initialise the font
        family = 'Helvetica'
        self.fonts = {
            'small': Font(family=family, name='small', size=SMALL_FONT, weight='normal', slant='roman'),
            'small_i': Font(family=family, name='small_i', size=SMALL_FONT, weight='normal', slant='italic'),
            'small_b': Font(family=family, name='small_b', size=SMALL_FONT, weight='bold', slant='roman'),
            'small_ib': Font(family=family, name='small_ib', size=SMALL_FONT, weight='bold', slant='italic'),
            'medium': Font(family=family, name='medium', size=MEDIUM_FONT, weight='normal', slant='roman'),
            'medium_i': Font(family=family, name='medium_i', size=MEDIUM_FONT, weight='normal', slant='italic'),
            'medium_b': Font(family=family, name='medium_b', size=MEDIUM_FONT, weight='bold', slant='roman'),
            'large': Font(family=family, name='large', size=LARGE_FONT, weight='normal', slant='roman'),
            'large_i': Font(family=family, name='large_i', size=LARGE_FONT, weight='normal', slant='italic'),
            'large_b': Font(family=family, name='large_b', size=LARGE_FONT, weight='bold', slant='roman')
        }
        # Initialise the style
        self.style.configure('.', font=self.fonts['small'])
        # Labels
        self.style.configure('TLabel', anchor=tk.W, font=self.fonts['small'])
        self.style.configure('primary.Inverse.TLabel', anchor=tk.CENTER, justify=tk.CENTER,
                             font=self.fonts['large'])
        self.style.configure('secondary.TLabelframe.Label', anchor=tk.CENTER, justify=tk.CENTER,
                             foreground=self.style.colors.secondary, font=self.fonts['medium'])
        # Buttons
        self.style.configure('TButton', anchor=tk.CENTER, padding=PADDING, font=self.fonts['medium_i'])
        self.style.configure('primary.TButton', anchor=tk.CENTER, padding=PADDING, font=self.fonts['large_i'])
        self.style.configure('secondary.Outline.TButton', anchor=tk.CENTER, padding=PADDING,
                             font=self.fonts['medium_i'])
        self.style.configure('Small.secondary.Outline.TButton', anchor=tk.CENTER, padding=PADDING,
                             font=self.fonts['small_i'])
        # Entries
        self.style.configure('TCombobox', padding=PADDING, font=self.fonts['small_b'])
        self.style.configure('TSpinbox', padding=PADDING, font=self.fonts['small_b'])
        self.style.configure('TEntry', padding=PADDING, font=self.fonts['small_b'])
        # Notebook
        self.style.configure('TNotebook', bordercolor='white', borderwidth=0, padding=0, relief='flat',
                             tabmargins=0, highlightthickness=0, darkcolor='white', lightcolor='white')
        self.style.configure('TNotebook.Tab', padding=0, font=self.fonts['medium'], borderwidth=0,
                             relief='flat', tabmargins=0, darkcolor='white', lightcolor='white', highlightthickness=0)
        self.style.map('TNotebook.Tab', background=[('selected', self.style.colors.primary)],
                       foreground=[('selected', 'white')])
        # Treeview
        self.style.configure('primary.Treeview', fieldbackground='#FFFFFF', background='#FFFFFF', relief='flat',
                             font=self.fonts['small'], indent=0, rowheight=self.fonts['small'].metrics('linespace') + 2,
                             borderwidth=0, lightcolor=self.style.colors.primary, darkcolor=self.style.colors.primary)
        self.style.configure('primary.Treeview.Heading', relief='flat', font=self.fonts['small_i'],
                             background=self.style.colors.primary, darkcolor='white', lightcolor='white')
        self.style.configure('primary.Treeview.Item', indicatormargins=0, indicatorsize=0, padding=0)
        self.style.configure('primary.Treeview.Cell', padding=0)
        self.style.map('primary.Treeview', foreground=[('selected', self.style.colors.primary)],
                       background=[], relief=[('focus', 'flat'), ('!focus', 'flat')],
                       font=[('selected', self.fonts['small_ib'])])
        self.style.map('primary.Treeview.Heading', relief=[('active', 'sunken'), ('pressed', 'sunken')])
        # Scrollbar
        self.style.configure('primary.Vertical.TScrollbar', background='white', darkcolor='white',
                             lightcolor='white', bordercolor=self.style.colors.primary, troughrelief='flat',
                             relief='flat', troughcolor='white', borderwidth=1, groovewidth=0, width=0)
        self.style.configure('secondary.Vertical.TScrollbar', background='white', darkcolor='white',
                             lightcolor='white', bordercolor=self.style.colors.secondary, troughrelief='flat',
                             relief='flat', troughcolor='white', borderwidth=1, groovewidth=0, width=0)

        # Misc
        self.style.configure('Horizontal.TFloodgauge', thickness=30, barsize=60)
        # Matplotlib
        mplstyle.use('fast')
        rc('font', size=SMALL_FONT, family='sans-serif')
        rc('grid', color='0.8', linewidth=0.5)
        rc('path', simplify=True, simplify_threshold=0.111111111111)
        rc('agg.path', chunksize=200000)
        rc('lines', antialiased=True)

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

        # Main layout
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
        self.windows['left'].columnconfigure(0, weight=1, minsize=self.width // 6)
        self.windows['left'].rowconfigure(0, weight=1)
        self.windows['load'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0), padx=(WINDOW_PADDING, 0))
        self.windows['left'].rowconfigure(1, weight=2)
        self.windows['analyse'].grid(row=1, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=(WINDOW_PADDING, 0))
        # Center layout
        self.windows['center'].columnconfigure(0, weight=1)
        self.windows['center'].rowconfigure(0, weight=1)
        self.windows['analysis_view'].grid(row=0, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=WINDOW_PADDING)
        # Right layout
        self.windows['right'].columnconfigure(0, weight=1, minsize=self.width // 6)
        self.windows['right'].rowconfigure(0, weight=1)
        self.windows['results'].grid(row=0, column=0, sticky="nsew", pady=WINDOW_PADDING, padx=(0, WINDOW_PADDING))

    def init_binds(self):
        self.bind('<Control-b>', lambda event: self.load_trace())
        self.bind('<Control-e>', lambda event: self.initial_estimate())
        self.bind('<Control-i>', lambda event: self.iterate())
        self.bind('<Control-s>', lambda event: self.save_results())
        self.bind('<o>', lambda event: self.toggle_analysis_view_options())

    def init_tooltips(self):
        tooltip.Hovertip(self.widgets['load']['browse'], "Ctrl + B", hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['estimate'], "Ctrl + E", hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['iterate'], "Ctrl + I", hover_delay=300)
        tooltip.Hovertip(self.widgets['results']['save'], "Ctrl + S", hover_delay=300)
        tooltip.Hovertip(self.widgets['analysis_view']['options_toggle'], "O", hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['bounds_input'],
                         "Median filtering window size for bounding events", hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['estimate_cutoff_input'], 'Cutoff frequency for the estimate',
                         hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['threshold_window_input'], 'Window size for threshold calculation'
                                                                            ' (estimate and iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['z_score_input'], 'Z-score for threshold (estimate and iterate)',
                         hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['cutoff_input'], 'Cutoff frequency for the iteration',
                         hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['event_direction_input'], 'Initial iteration event direction',
                         hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['replace_factor_input'], 'Factor for replacing events '
                                                                          '(multiple of event width)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['replace_gap_input'], 'Gap for replacing events '
                                                                       '(multiple of event width)', hover_delay=300)
        tooltip.Hovertip(self.widgets['load']['units'], 'Amplitude: pA, Duration: ms, Area: pC', hover_delay=300)

    def init_landing_page(self):
        # Landing page
        self.widgets['landing'] = dict()
        self.widgets['landing']['main'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')
        self.widgets['landing']['main'].grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.widgets['landing']['inner'] = ttk.Frame(self.widgets['landing']['main'], style='primary.TFrame')
        self.widgets['landing']['main'].columnconfigure(0, weight=1)
        self.widgets['landing']['main'].rowconfigure(0, weight=1)
        self.widgets['landing']['inner'].grid(row=0, column=0, sticky="nsew", padx=WINDOW_PADDING, pady=WINDOW_PADDING)

        # Information frame
        self.widgets['landing']['information'] = ttk.Frame(self.widgets['landing']['inner'])
        # Browse button
        self.widgets['landing']['button'] = ttk.Button(self.widgets['landing']['inner'], text="Browse", underline=0,
                                                       command=self.landing_load)

        # Layout
        self.widgets['landing']['inner'].columnconfigure(0, weight=1)
        self.widgets['landing']['inner'].rowconfigure(0, weight=1)
        self.widgets['landing']['information'].grid(row=0, column=0, sticky="nsew", padx=WINDOW_PADDING,
                                                    pady=(WINDOW_PADDING, 0))
        self.widgets['landing']['button'].grid(row=1, column=0, sticky="ew")
        self.widgets['landing']['button'].focus_set()

        # Left frame
        self.widgets['landing']['left'] = ttk.Frame(self.widgets['landing']['information'])
        # Logo
        logo = tk.PhotoImage(file=os.path.join(self.data_path, 'logo.png')).subsample(2, 2)
        self.widgets['landing']['image'] = tk.Label(self.widgets['landing']['information'], image=logo)
        self.widgets['landing']['image'].image = logo
        # Right frame
        self.widgets['landing']['right'] = ttk.Frame(self.widgets['landing']['information'])

        # Layout
        self.widgets['landing']['information'].rowconfigure(0, weight=1)
        self.widgets['landing']['information'].columnconfigure(0, minsize=185)
        self.widgets['landing']['left'].grid(row=0, column=0, sticky="nsew")
        self.widgets['landing']['information'].columnconfigure(1, weight=1)
        self.widgets['landing']['image'].grid(row=0, column=1, sticky="nsew")
        self.widgets['landing']['information'].columnconfigure(2, minsize=185)
        self.widgets['landing']['right'].grid(row=0, column=2, sticky="nsew")

        # Left information
        self.widgets['landing']['left_gap'] = ttk.Label(self.widgets['landing']['left'], text="")
        email_icon = tk.PhotoImage(file=os.path.join(self.data_path, 'email_icon.png'))
        self.widgets['landing']['email_icon'] = ttk.Label(self.widgets['landing']['left'], text="", image=email_icon)
        self.widgets['landing']['email_icon'].image = email_icon
        self.widgets['landing']['email'] = ttk.Label(self.widgets['landing']['left'], text="el20dlgc@leeds.ac.uk",
                                                     foreground='blue', cursor='hand2')
        self.widgets['landing']['email'].bind("<Button-1>", lambda event: open_new("mailto:el20dlgc@leeds.ac.uk"))

        # Layout
        self.widgets['landing']['left'].rowconfigure(0, weight=1)
        self.widgets['landing']['left'].columnconfigure(1, weight=1)
        self.widgets['landing']['left_gap'].grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.widgets['landing']['email_icon'].grid(row=1, column=0, sticky="nsew", padx=PADDING, pady=(0, PADDING))
        self.widgets['landing']['email'].grid(row=1, column=1, sticky="nsew", pady=(0, PADDING))

        # Right information
        self.widgets['landing']['right_gap_1'] = ttk.Label(self.widgets['landing']['right'], text="")
        self.widgets['landing']['right_gap_2'] = ttk.Label(self.widgets['landing']['right'], text="")
        github_icon = tk.PhotoImage(file=os.path.join(self.data_path, 'github_icon.png'))
        self.widgets['landing']['github_icon'] = ttk.Label(self.widgets['landing']['right'], text="", image=github_icon)
        self.widgets['landing']['github_icon'].image = github_icon
        self.widgets['landing']['github'] = ttk.Label(self.widgets['landing']['right'], text="notnormal",
                                                      foreground='blue', cursor='hand2')
        self.widgets['landing']['github'].bind("<Button-1>", lambda event:
        open_new("https://www.github.com/el20dlgc/notnormal"))

        # Layout
        self.widgets['landing']['right'].rowconfigure(0, weight=1)
        self.widgets['landing']['right'].columnconfigure(0, weight=1)
        self.widgets['landing']['right_gap_1'].grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.widgets['landing']['right_gap_2'].grid(row=1, column=0, sticky="nsew")
        self.widgets['landing']['github_icon'].grid(row=1, column=1, sticky="nsew", padx=(0, PADDING), pady=(0, PADDING))
        self.widgets['landing']['github'].grid(row=1, column=2, sticky="nsew", padx=(0, PADDING), pady=(0, PADDING))

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

        # Path information
        self.trace_information['path'] = tk.StringVar()
        self.trace_information['path'].set('')
        # Filename information
        self.widgets['load']['filename_label'] = ttk.Label(self.windows['load_internal'], text="Filename")
        self.trace_information['filename'] = tk.StringVar()
        self.trace_information['filename'].set('')
        wrap = (self.width // 6) - (7 * PADDING) - self.fonts['small'].measure('Sample Rate (Hz)')
        self.widgets['load']['filename'] = ttk.Label(self.windows['load_internal'],
                                                     textvariable=self.trace_information['filename'], wraplength=wrap)
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
        self.widgets['load']['duration_separator'] = ttk.Separator(self.windows['load_internal'])
        # Units
        self.widgets['load']['units_label'] = ttk.Label(self.windows['load_internal'], text="Units")
        self.trace_information['units'] = tk.StringVar()
        self.trace_information['units'].set('pA')
        self.widgets['load']['units'] = ttk.Combobox(
            self.windows['load_internal'],
            values=['A', 'mA', 'µA', 'nA', 'pA', 'fA'],
            textvariable=self.trace_information['units'],
            state='readonly',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.widgets['load']['units'].bind('<<ComboboxSelected>>', lambda _: self.update_units())
        # Browse button
        self.widgets['load']['browse'] = ttk.Button(self.windows['load_internal'], text="Browse", underline=0,
                                                    command=self.load_trace, style='secondary.Outline.TButton')

        # Filename layout
        self.windows['load_internal'].columnconfigure(0, weight=1)
        self.windows['load_internal'].columnconfigure(1, weight=2)
        self.windows['load_internal'].rowconfigure(0, weight=1)
        self.widgets['load']['filename_label'].grid(row=0, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['filename'].grid(row=0, column=1, sticky="e", padx=PADDING)
        self.widgets['load']['filename_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        # Sample rate layout
        self.windows['load_internal'].rowconfigure(2, weight=1)
        self.widgets['load']['sample_rate_label'].grid(row=2, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['sample_rate'].grid(row=2, column=1, sticky="e", padx=PADDING)
        self.widgets['load']['sample_rate_separator'].grid(row=3, column=0, columnspan=2, sticky="nsew")
        # Samples layout
        self.windows['load_internal'].rowconfigure(4, weight=1)
        self.widgets['load']['samples_label'].grid(row=4, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['samples'].grid(row=4, column=1, sticky="e", padx=PADDING)
        self.widgets['load']['samples_separator'].grid(row=5, column=0, columnspan=2, sticky="nsew")
        # Duration layout
        self.windows['load_internal'].rowconfigure(6, weight=1)
        self.widgets['load']['duration_label'].grid(row=6, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['duration'].grid(row=6, column=1, sticky="e", padx=PADDING)
        self.widgets['load']['duration_separator'].grid(row=7, column=0, columnspan=2, sticky="nsew")
        # Theme layout
        self.windows['load_internal'].rowconfigure(8, weight=1)
        self.widgets['load']['units_label'].grid(row=8, column=0, sticky="nsew", padx=PADDING)
        self.widgets['load']['units'].grid(row=8, column=1, sticky="e", padx=PADDING)
        # Browse layout
        self.widgets['load']['browse'].grid(row=9, column=0, columnspan=2, sticky="nsew", padx=PADDING,
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
            format='%.0f',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['bounds_filter'] = self.analysis_options['bounds_filter'].get()
        self.widgets['analyse']['bounds_separator'] = ttk.Separator(self.windows['analyse_internal'])
        self.widgets['analyse']['bounds_input'].configure(font=self.fonts['small'])
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
        self.widgets['analyse']['estimate_cutoff_input'].configure(font=self.fonts['small'])
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
        self.widgets['analyse']['threshold_window_input'].configure(font=self.fonts['small'])
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
        self.widgets['analyse']['z_score_input'].configure(font=self.fonts['small'])
        # Estimate button
        self.widgets['analyse']['estimate'] = ttk.Button(
            self.windows['analyse_internal'],
            text="Estimate",
            underline=0,
            command=self.initial_estimate,
            style='secondary.Outline.TButton'
        )
        # Cutoff input
        self.analysis_options['cutoff'] = tk.DoubleVar()
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
        self.widgets['analyse']['cutoff_input'].configure(font=self.fonts['small'])
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
        self.widgets['analyse']['event_direction_input'].configure(font=self.fonts['small'])
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
            format='%.0f',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['replace_factor'] = self.analysis_options['replace_factor'].get()
        self.widgets['analyse']['replace_factor_separator'] = ttk.Separator(self.windows['analyse_internal'])
        self.widgets['analyse']['replace_factor_input'].configure(font=self.fonts['small'])
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
            format='%.0f',
            width=ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['replace_gap'] = self.analysis_options['replace_gap'].get()
        self.widgets['analyse']['replace_gap_input'].configure(font=self.fonts['small'])
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
        # Internal frame for figure
        self.windows['analysis_view_figure'] = ttk.Frame(self.windows['analysis_view'])
        # Layout
        self.windows['analysis_view'].columnconfigure(0, weight=1)
        self.widgets['analysis_view']['title'].grid(row=0, column=0, sticky="nsew", pady=(WINDOW_PADDING, 0))
        self.windows['analysis_view'].rowconfigure(1, weight=3)
        self.windows['analysis_view_figure'].grid(row=1, column=0, sticky="nsew", padx=WINDOW_PADDING,
                                                  pady=WINDOW_PADDING)

        # Create figure, axis and canvas
        self.widgets['analysis_view']['fig'] = Figure(layout='constrained')
        ax = self.widgets['analysis_view']['fig'].add_subplot(111)
        ax.set_xlabel("Time ($s$)", fontsize=MEDIUM_FONT)
        ax.set_ylabel("Current" + f" (${self.trace_information['units'].get()}$)", fontsize=MEDIUM_FONT)
        # Create canvas
        self.widgets['analysis_view']['canvas'] = CustomFigureCanvas(self.widgets['analysis_view']['fig'],
                                                                    self.windows['analysis_view_figure'], self)
        self.widgets['analysis_view']['canvas'].draw()
        # Create toolbar
        self.widgets['analysis_view']['toolbar'] = NavigationToolbar2Tk(self.widgets['analysis_view']['canvas'],
                                                                        self.windows['analysis_view_figure'],
                                                                        pack_toolbar=False)
        for child in self.widgets['analysis_view']['toolbar'].winfo_children():
            child.pack_forget()
            child.configure(background='white', border=1)
        for child in self.widgets['analysis_view']['toolbar'].winfo_children():
            if child != self.widgets['analysis_view']['toolbar'].children['!button4']:
                child.pack(side=tk.LEFT)
        # Reconfigure commands to downsample
        for name in ('Home', 'Back', 'Forward'):
            command = getattr(self.widgets['analysis_view']['toolbar'], name.lower())
            btn = self.widgets['analysis_view']['toolbar']._buttons[name]
            btn.configure(command=self.toolbar_wrapper(command))
        self.widgets['analysis_view']['toolbar'].update()

        # Progress bar
        self.widgets['analysis_view']['progress_bar'] = Floodgauge(
            self.windows['analysis_view_figure'],
            orient=tk.HORIZONTAL,
            mode='indeterminate',
            length=180,
            style='Horizontal.TFloodgauge'
        )
        # Toolbar separator
        self.widgets['analysis_view']['toolbar_separator'] = ttk.Separator(self.windows['analysis_view_figure'])

        # Figure frame layout
        self.windows['analysis_view_figure'].columnconfigure(0, weight=2)
        self.widgets['analysis_view']['toolbar'].grid(row=0, column=0, sticky="nsew", padx=PADDING)
        self.widgets['analysis_view']['progress_bar'].grid(row=0, column=1, sticky='nse', padx=PADDING, pady=PADDING)
        self.widgets['analysis_view']['toolbar_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.windows['analysis_view_figure'].rowconfigure(2, weight=1)
        self.widgets['analysis_view']['canvas'].get_tk_widget().grid(row=2, column=0,  columnspan=2, sticky="nsew",
                                                                     padx=PADDING, pady=(PADDING, 0))

        # Slider figure frame
        self.windows['analysis_view_slider'] = ttk.Frame(self.windows['analysis_view_figure'])
        self.windows['analysis_view_slider'].grid(row=3, column=0, columnspan=2, sticky="nsew", padx=PADDING,
                                                  pady=(0, PADDING))
        # Create figure, axis and canvas
        self.widgets['analysis_view']['slider_fig'] = Figure(figsize=(50, 0.25))
        ax = self.widgets['analysis_view']['slider_fig'].add_subplot(111)
        self.widgets['analysis_view']['slider_fig'].subplots_adjust(
            left=0.02,
            right=0.98,
            top=1,
            bottom=0,
            wspace=0,
            hspace=0
        )
        # Create canvas
        self.widgets['analysis_view']['slider_canvas'] = FigureCanvasTkAgg(self.widgets['analysis_view']['slider_fig'],
                                                                           self.windows['analysis_view_slider'])
        self.widgets['analysis_view']['slider_canvas'].draw()
        # Add the range slider
        handles = dict(facecolor='white', edgecolor=self.style.colors.secondary, size=10)
        self.widgets['analysis_view']['slider'] = RangeSlider(
            ax,'',0.0,1.0, closedmax=False, closedmin=False,
          facecolor=self.style.colors.primary, handle_style=handles)
        self.widgets['analysis_view']['slider'].valtext.set_visible(False)
        self.widgets['analysis_view']['slider'].on_changed(self.schedule_update_figure_xlimits)
        # Layout
        self.windows['analysis_view_slider'].columnconfigure(0, weight=1)
        self.windows['analysis_view_slider'].rowconfigure(0, weight=1)
        self.widgets['analysis_view']['slider_canvas'].get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Bind events
        self.widgets['analysis_view']['canvas'].mpl_connect("key_press_event",
                                                            partial(self.toolbar_key_press, 'analysis_view'))
        self.widgets['analysis_view']['canvas'].mpl_connect("draw_event", lambda _: self.after(100, self.update_slider))
        self.widgets['analysis_view']['canvas'].mpl_connect("button_release_event", lambda _: self.update_figure_xlimits())
        self.widgets['analysis_view']['canvas'].mpl_connect("pick_event", self.pick_event)

    def init_slider(self):
        if type(self.time_vector) is not np.ndarray:
            return

        limits = (self.time_vector[0], self.time_vector[-1])
        padding = (limits[1] - limits[0]) / 20
        limits = (limits[0] - padding, limits[1] + padding)
        slider = self.widgets['analysis_view']['slider']
        slider.eventson = False
        slider.valmin = limits[0]
        slider.valmax = limits[1]
        slider.valinit = limits
        slider.valstep = self.time_step
        slider.set_val(limits)
        slider.ax.set_xlim(limits)
        slider.eventson = True

    def init_analysis_view_options(self):
        # Show/hide options button as title
        self.widgets['analysis_view']['options_toggle'] = ttk.Button(
            self.windows['analysis_view'],
            text="Options",
            underline=0,
            command=self.toggle_analysis_view_options,
            style='primary.TButton'
        )
        # Internal frame for options
        self.windows['analysis_view_options'] = ttk.Frame(self.windows['analysis_view'])
        # Layout
        self.widgets['analysis_view']['options_toggle'].grid(row=2, column=0, sticky="ew")
        self.windows['analysis_view'].rowconfigure(3, weight=1)
        self.windows['analysis_view_options'].grid(row=3, column=0, sticky="nsew", padx=WINDOW_PADDING,
                                                   pady=(0, WINDOW_PADDING))

        # Reset labelframe
        self.widgets['analysis_view']['reset'] = ttk.LabelFrame(self.windows['analysis_view_options'], text="Reset",
                                                                style='secondary.TLabelframe', labelanchor='n')
        # Lines labelframe
        self.widgets['analysis_view']['lines'] = ttk.LabelFrame(self.windows['analysis_view_options'], text="Lines",
                                                                style='secondary.TLabelframe', labelanchor='n')
        # Layout
        self.windows['analysis_view_options'].rowconfigure(0, weight=1)
        self.windows['analysis_view_options'].columnconfigure(0, weight=0)
        self.windows['analysis_view_options'].columnconfigure(1, weight=4)
        self.widgets['analysis_view']['reset'].grid(row=0, column=0, sticky="nsew", padx=(PADDING, 0), pady=PADDING)
        self.widgets['analysis_view']['lines'].grid(row=0, column=1, sticky="nsew", padx=PADDING, pady=PADDING)

        # Reset labelframe options
        items = [('all', self.reset_all), ('algorithm', self.reset_algorithm), ('figure', self.reset_figure),
                 ('results', self.reset_results)]
        self.widgets['analysis_view']['reset'].columnconfigure(0, weight=1, uniform='options')
        for i, item in enumerate(items):
            self.widgets['analysis_view'][f'reset_{item[0]}_button'] = ttk.Button(
                self.widgets['analysis_view']['reset'],
                text=item[0].capitalize(),
                width=2 * ENTRY_WIDTH,
                command=item[1],
                style='Small.secondary.Outline.TButton'
            )
            self.widgets['analysis_view']['reset'].rowconfigure(i, weight=1, uniform='options')
            self.widgets['analysis_view'][f'reset_{item[0]}_button'].grid(row=i, column=0, padx=PADDING)

        # Lines labelframe options
        keys = [('trace', 'Trace'), ('baseline', 'Baseline'), ('threshold', 'Threshold'),
                ('calculation_trace', 'Calculation Trace'), ('filtered_trace', 'Filtered Trace'), ('events', 'Events')]
        options = [('show', 'bool', self.update_figure_show), ('linewidth', 'float', self.update_figure_linewidth),
                   ('colour', 'colour', self.update_figure_color), ('style', 'str', self.update_figure_style)]
        # Fixed options
        self.figure_options['zorder'] = dict(trace=2, calculation_trace=3, filtered_trace=4, baseline=5, threshold=6,
                                             events=7)

        # Initialise the headings
        self.widgets['analysis_view']['lines'].rowconfigure(0, weight=1, uniform='options')
        for i, option in enumerate(options):
            # Initialise the dictionaries
            self.figure_options[option[0]] = dict()
            self.default_figure_options[option[0]] = dict()
            self.widgets['analysis_view'][option[0]] = dict()
            # Initialise the headings
            self.widgets['analysis_view'][f'{option[0]}_label'] = ttk.Label(self.widgets['analysis_view']['lines'],
                                                                            text=option[0].capitalize(), anchor='center')
            self.widgets['analysis_view']['lines'].columnconfigure(i + 1, weight=1, uniform='options')
            self.widgets['analysis_view'][f'{option[0]}_label'].grid(row=0, column=i + 1, sticky="nsew", padx=PADDING)
        # Heading separator
        self.widgets['analysis_view']['heading_separator'] = ttk.Separator(self.widgets['analysis_view']['lines'])
        self.widgets['analysis_view']['heading_separator'].grid(row=1, column=0, columnspan=len(options) + 1,
                                                                sticky="nsew")

        # Initialise the rows
        self.widgets['analysis_view']['lines'].columnconfigure(0, weight=1, uniform='options')
        i = 2
        for key in keys:
            self.widgets['analysis_view'][f'{key[0]}_label'] = ttk.Label(self.widgets['analysis_view']['lines'],
                                                                      text=key[1])
            self.widgets['analysis_view']['lines'].rowconfigure(i, weight=1, uniform='options')
            self.widgets['analysis_view'][f'{key[0]}_label'].grid(row=i, column=0, sticky="nsew", padx=PADDING)
            if i < (2 * len(keys)) + 1:
                self.widgets['analysis_view'][f'{key[0]}_separator'] = ttk.Separator(self.widgets['analysis_view']['lines'])
                self.widgets['analysis_view'][f'{key[0]}_separator'].grid(row=i + 1, column=0,
                                                                          columnspan=len(options) + 1, sticky="nsew")
            # Options for each key
            j = 1
            for option in options:
                # General configuration
                if option[1] == 'bool':
                    self.figure_options[option[0]][key[0]] = tk.BooleanVar()
                    self.widgets['analysis_view'][option[0]][key[0]] = ttk.Checkbutton(
                        self.widgets['analysis_view']['lines'],
                        variable=self.figure_options[option[0]][key[0]],
                        command=partial(option[2],key[0]),
                        style='Roundtoggle.Toolbutton',
                    )
                elif option[1] == 'float':
                    self.figure_options[option[0]][key[0]] = tk.DoubleVar()
                    self.widgets['analysis_view'][option[0]][key[0]] = ttk.Spinbox(
                        self.widgets['analysis_view']['lines'],
                        textvariable=self.figure_options[option[0]][key[0]],
                        format='%2.1f',
                        command=partial(option[2], key[0]),
                        justify='center',
                        width=ENTRY_WIDTH
                    )
                    self.widgets['analysis_view'][option[0]][key[0]].bind("<Return>", lambda _: partial(option[2], key[0]))
                    self.widgets['analysis_view'][option[0]][key[0]].bind("<FocusOut>", lambda _: partial(option[2], key[0]))
                    self.widgets['analysis_view'][option[0]][key[0]].configure(font=self.fonts['small'])
                elif option[1] == 'colour':
                    self.figure_options[option[0]][key[0]] = tk.StringVar()
                    self.widgets['analysis_view'][option[0]][key[0]] = ttk.Button(
                        self.widgets['analysis_view']['lines'],
                        command=partial(option[2],key[0]),
                        width=ENTRY_WIDTH - 3,
                    )
                elif option[1] == 'str':
                    self.figure_options[option[0]][key[0]] = tk.StringVar()
                    self.widgets['analysis_view'][option[0]][key[0]] = ttk.Combobox(
                        self.widgets['analysis_view']['lines'],
                        textvariable=self.figure_options[option[0]][key[0]],
                        state='readonly',
                        justify='center',
                        width=ENTRY_WIDTH
                    )
                    self.widgets['analysis_view']['style'][key[0]].bind('<<ComboboxSelected>>', lambda _, x = key[0]:
                                                                        option[2](x))
                    self.widgets['analysis_view'][option[0]][key[0]].configure(font=self.fonts['small'])

                # Specific configuration
                if option[0] == 'show':
                    self.figure_options[option[0]][key[0]].set(True if key[0] in ['trace', 'baseline', 'threshold'] else
                                                               False)
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=(PADDING + 3, PADDING),
                                                                          pady=(1, 0))
                elif option[0] == 'linewidth':
                    self.figure_options[option[0]][key[0]].set(1.0)
                    self.widgets['analysis_view'][option[0]][key[0]].configure(from_=0, to=4, increment=0.1)
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=PADDING)
                elif option[0] == 'colour':
                    self.figure_options[option[0]][key[0]].set(self.colours[key[0]])
                    self.style.configure(f'{key[0]}_colour.TButton', background=self.colours[key[0]],
                                         borderwidth=0)
                    self.style.map(f'{key[0]}_colour.TButton', background=[('active', self.colours[key[0]])])
                    self.widgets['analysis_view'][option[0]][key[0]].configure(style=f'{key[0]}_colour.TButton')
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=PADDING)
                elif option[0] == 'style':
                    self.figure_options[option[0]][key[0]].set('-')
                    self.widgets['analysis_view'][option[0]][key[0]].configure(values=['-', '--', ':', '-.'])
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=PADDING)


                self.default_figure_options[option[0]][key[0]] = self.figure_options[option[0]][key[0]].get()
                j += 1

            i += 2

        # Add grid to the top left corner
        self.figure_options['grid'] = tk.BooleanVar(value=False)
        self.default_figure_options['grid'] = self.figure_options['grid'].get()
        self.widgets['analysis_view']['grid'] = ttk.Button(
            self.widgets['analysis_view']['lines'],
            text='Grid',
            width=ENTRY_WIDTH,
            command=self.update_figure_grid,
            style='Small.secondary.Outline.TButton',
        )
        self.widgets['analysis_view']['grid'].grid(row=0, column=0, sticky="nw", padx=PADDING + 2)

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
        # Feature window
        self.widgets['results']['feature_window'] = None

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
        self.trace = None
        self.time_vector = None
        self.time_step = None
        # Clear the trace information
        for key in self.trace_information:
            self.trace_information[key].set('')
        self.trace_information['units'].set('pA')
        # Set the flags
        self.flags['loaded'] = False

    def reset_algorithm(self):
        for key in self.default_analysis_options:
            self.analysis_options[key].set(self.default_analysis_options[key])
        # Disable iterate options
        self.toggle_iterate_options(['disabled'])

    def reset_figure(self):
        # Reset the figure options
        for key in self.default_figure_options:
            if isinstance(self.default_figure_options[key], dict):
                for sub_key in self.default_figure_options[key]:
                    self.figure_options[key][sub_key].set(self.default_figure_options[key][sub_key])
                    if key == 'colour':
                        colour = self.default_figure_options[key][sub_key]
                        self.style.configure(
                            f'{sub_key}_colour.TButton',
                            background=colour,
                            bordercolor=colour,
                            lightcolor=colour,
                            darkcolor=colour
                        )
                        self.style.map(
                            f'{sub_key}_colour.TButton',
                            background=[('active', colour)],
                            bordercolor=[('active', colour)],
                            lightcolor=[('active', colour)],
                            darkcolor=[('active', colour)]
                        )
                        self.widgets['analysis_view']['colour'][sub_key].configure(style=f'{sub_key}_colour.TButton')
            else:
                self.figure_options[key].set(self.default_figure_options[key])
        # Update the figure
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

    def update_load(self, path):
        self.trace_information['path'].set(path)

        if path != '':
            if len(basename(path)) > 63:
                self.trace_information['filename'].set(f'{basename(path)[0:60]}...')
            else:
                self.trace_information['filename'].set(basename(path))

        if self.trace is not None:
            self.trace_information['samples'].set(f'{len(self.trace):.0f}')

        if self.time_vector is not None:
            self.trace_information['sample_rate'].set(f'{1 / self.time_step:.0f}')

        if self.time_vector is not None:
            self.trace_information['duration'].set(f'{self.time_vector[-1] - self.time_vector[0]:.1f}')

    def update_units(self):
        ax = self.widgets['analysis_view']['fig'].axes[0]
        units = self.trace_information['units'].get()
        ax.set_ylabel("Current" + f' (${units}$)', fontsize=MEDIUM_FONT)

        if units == 'A':
            tooltip.Hovertip(self.widgets['load']['units'], f'Amplitude: A, Duration: ms, Area: C',
                             hover_delay=300)
        else:
            tooltip.Hovertip(self.widgets['load']['units'], f'Amplitude: {units[0]}A, Duration: ms, Area: {units[0]}C',
                             hover_delay=300)
        # Redraw
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    @cython.boundscheck(False)
    def update_figure(self, title: str = None, retain_view: bool = True):
        if self.widgets['analysis_view']['fig'].axes:
            ax = self.widgets['analysis_view']['fig'].axes[0]
        else:
            return

        if type(self.time_vector) is np.ndarray and self.time_step:
            # Clear lines except the trace
            for line in ax.get_lines():
                    line.remove()

            # Get xlim
            xlim = ax.get_xlim() if retain_view else (self.time_vector[0], self.time_vector[-1])

            # Plot the trace
            if type(self.trace) is np.ndarray and self.figure_options['show']['trace'].get():
                ax.plot(*self.downsample_line(self.time_vector, self.trace, xlim), label='trace')

            # Plot the calculation trace
            if type(self.calculation_trace) is np.ndarray and self.figure_options['show']['calculation_trace'].get():
                ax.plot(*self.downsample_line(self.time_vector, self.calculation_trace, xlim), label='calculation_trace')

            # Plot the filtered trace
            if type(self.filtered_trace) is np.ndarray and self.figure_options['show']['filtered_trace'].get():
                ax.plot(*self.downsample_line(self.time_vector, self.filtered_trace, xlim), label='filtered_trace')

            # Plot the baseline
            if type(self.baseline) is np.ndarray and self.figure_options['show']['baseline'].get():
                ax.plot(*self.downsample_line(self.time_vector, self.baseline, xlim), label='baseline')

            # Plot the threshold
            if (type(self.threshold) is np.ndarray and type(self.baseline) is np.ndarray and
                    self.figure_options['show']['threshold'].get()):
                ax.plot(*self.downsample_line(self.time_vector, self.baseline + self.threshold, xlim), label='threshold')
                ax.plot(*self.downsample_line(self.time_vector, self.baseline - self.threshold, xlim), label='threshold')

            # Plot the events
            if type(self.event_coordinates) is np.ndarray and self.figure_options['show']['events'].get():
                ax.plot(*self.downsample_events(xlim), label='events', picker=True, pickradius=5)

            # Set visibility, colour, linewidth and linestyle
            for line in ax.get_lines():
                if line.get_label() not in self.colours.keys():
                    continue
                line.set_color(self.figure_options['colour'][line.get_label()].get())
                line.set_linewidth(self.figure_options['linewidth'][line.get_label()].get())
                line.set_linestyle(self.figure_options['style'][line.get_label()].get())
                line.set_zorder(self.figure_options['zorder'][line.get_label()])

        else:
            ax.clear()

        # Set limits
        if not retain_view:
            ax.relim()
            ax.autoscale(True)
            self.widgets['analysis_view']['toolbar'].update()
            self.widgets['analysis_view']['toolbar'].push_current()
            self.update_slider()

        # Set title
        if title:
            ax.set_title(title, fontsize=LARGE_FONT)

        # Set labels and grid
        ax.set_xlabel("Time (s)", fontsize=MEDIUM_FONT)
        ax.set_ylabel("Current" + f" (${self.trace_information['units'].get()}$)", fontsize=MEDIUM_FONT)
        ax.grid(self.figure_options['grid'].get())
        self.widgets['analysis_view']['canvas'].draw_idle()
        self.widgets['analysis_view']['canvas'].flush_events()

    def update_figure_grid(self):
        grid = self.figure_options['grid'].get()
        self.figure_options['grid'].set(not grid)
        self.widgets['analysis_view']['fig'].axes[0].grid(not grid)
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def update_figure_show(self, key):
        # Check if the figure is initialised
        if self.widgets['analysis_view']['fig'].axes:
            ax = self.widgets['analysis_view']['fig'].axes[0]
        else:
            return

        # Check if time vector and time step are initialised
        if type(self.time_vector) is not np.ndarray or not self.time_step:
            return

        show = self.figure_options['show'][key].get()
        if not show:
            lines = ax.get_lines()
            for line in lines:
                if key == line.get_label():
                    line.remove()
        else:
            xlim = ax.get_xlim()
            lines = []
            if key == 'trace' and type(self.trace) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, self.trace, xlim), label='trace'))
            elif key == 'calculation_trace' and type(self.calculation_trace) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, self.calculation_trace, xlim), label='calculation_trace'))
            elif key == 'filtered_trace' and type(self.filtered_trace) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, self.filtered_trace, xlim), label='filtered_trace'))
            elif key == 'baseline' and type(self.baseline) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, self.baseline, xlim), label='baseline'))
            elif key == 'threshold' and type(self.threshold) is np.ndarray and type(self.baseline) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, self.baseline + self.threshold, xlim), label='threshold'))
                lines.append(ax.plot(*self.downsample_line(self.time_vector, self.baseline - self.threshold, xlim), label='threshold'))
            elif key == 'events' and type(self.event_coordinates) is np.ndarray:
                lines.append(ax.plot(*self.downsample_events(xlim), label='events', picker=True, pickradius=5))
            # Set visibility, colour, linewidth and linestyle
            for line in lines:
                line = line[0]
                line.set_color(self.figure_options['colour'][line.get_label()].get())
                line.set_linewidth(self.figure_options['linewidth'][line.get_label()].get())
                line.set_linestyle(self.figure_options['style'][line.get_label()].get())
                line.set_zorder(self.figure_options['zorder'][line.get_label()])
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def update_figure_color(self, key):
        colour = colorchooser.askcolor(title="Choose color", initialcolor=self.figure_options['colour'][key].get())[1]
        if colour is None:
            return

        self.figure_options['colour'][key].set(colour)
        self.style.configure(f'{key}_colour.TButton', background=colour)
        self.style.map(f'{key}_colour.TButton', background=[('active', colour)])
        self.widgets['analysis_view']['colour'][key].configure(style=f'{key}_colour.TButton')

        if not self.widgets['analysis_view']['fig'].axes or not self.figure_options['show'][key].get():
            return

        lines = self.widgets['analysis_view']['fig'].axes[0].get_lines()
        for line in lines:
            if key == line.get_label():
                if colour == line.get_color():
                    continue
                line.set_color(colour)
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def update_figure_linewidth(self, key):
        if not self.widgets['analysis_view']['fig'].axes or not self.figure_options['show'][key].get():
            return

        width = self.figure_options['linewidth'][key].get()
        lines = self.widgets['analysis_view']['fig'].axes[0].get_lines()
        for line in lines:
            if key == line.get_label():
                if width == line.get_linewidth():
                    continue
                line.set_linewidth(width)
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def schedule_update_figure_xlimits(self, limits):
        if self.after_id is not None:
            self.after_cancel(self.after_id)

        self.after_id = self.after(50, self.update_figure_xlimits, limits)

    def update_figure_xlimits(self, limits = None):
        if limits:
            self.after_id = None
            ax = self.widgets['analysis_view']['fig'].axes[0]
            ax.set_xlim(limits)
            # Downsample all lines
            self.downsample_all()
            if np.all(np.round(limits, 5) == np.round(ax.get_xlim(), 5)):
                # Push state to toolbar
                self.widgets['analysis_view']['toolbar'].push_current()
        else:
            mode = self.widgets['analysis_view']['toolbar'].mode
            if mode not in (mode.ZOOM, mode.PAN):
                return
            # Downsample all lines
            self.downsample_all()
        # Redraw
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def update_slider(self):
        slider = self.widgets['analysis_view']['slider']
        limits = self.widgets['analysis_view']['fig'].axes[0].get_xlim()
        slider.eventson = False
        slider.set_val(limits)
        slider.eventson = True

    def update_figure_style(self, key):
        if not self.widgets['analysis_view']['fig'].axes or not self.figure_options['show'][key].get():
            return

        style = self.figure_options['style'][key].get()
        lines = self.widgets['analysis_view']['fig'].axes[0].get_lines()
        for line in lines:
            if key == line.get_label():
                if style == line.get_linestyle():
                    continue
                line.set_linestyle(style)
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

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
        # Feature window
        if self.widgets['results']['feature_window']:
            self.widgets['results']['feature_window'].destroy()

        if self.widgets['results']['notebook'].tabs():
            self.widgets['results']['notebook'].select(0)
            self.toggle_results('show')
        else:
            self.toggle_results('hide')

    def toggle_landing_page(self, toggle: str = None):
        if toggle is None and self.widgets['landing']['main'].winfo_manager():
            toggle = 'hide'
        elif toggle is None:
            toggle = 'show'

        if self.widgets['landing']['main'].winfo_manager() and toggle == 'hide':
            self.widgets['landing']['main'].grid_forget()
            self.geometry(f'{self.width}x{self.height}')
            self.state('zoomed')
            self.resizable(True, True)
            self.toggle_load('show')
            self.toggle_analyse('show')
            self.toggle_analysis_view('show')
            self.widgets['analyse']['estimate'].focus_set()
        elif toggle == 'show':
            # Hide all windows
            self.toggle_load('hide')
            self.toggle_analyse('hide')
            self.toggle_analysis_view('hide')
            self.toggle_analysis_view_options('hide')
            self.toggle_results('hide')
            self.widgets['landing']['main'].grid(row=0, column=0, sticky="nsew")
            self.geometry(f"700x400+{self.width // 2 - 350}+{self.height // 2 - 200}")
            self.state('normal')
            self.resizable(False, False)

    def toggle_load(self, toggle: str = None):
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

    def toggle_analyse(self, toggle: str = None):
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

    def toggle_iterate_options(self, toggle: list[str] = None):
        if toggle is None and self.widgets['analyse']['iterate'].instate(['!disabled']):
            toggle = ['disabled']
        elif toggle is None:
            toggle = ['!disabled']

        for key in ['iterate', 'replace_gap_input', 'replace_gap_label', 'replace_factor_input', 'replace_factor_label',
                    'event_direction_label', 'cutoff_input', 'cutoff_label']:
            self.widgets['analyse'][key].state(toggle)
        self.widgets['analyse']['event_direction_input'].state(['!readonly', toggle[0]] if toggle == ['disabled']
                                                               else [toggle[0], 'readonly'])

    def toggle_analysis_view(self, toggle: str = None):
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

    def toggle_analysis_view_options(self, toggle: str = None):
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

    def toggle_results(self, toggle: str = None):
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

    def create_feature_window(self):
        if self.flags['feature_win']:
            return
        self.flags['feature_win'] = True

        tab_id = self.widgets['results']['notebook'].select()
        text =  self.widgets['results']['notebook'].tab(tab_id, "text")
        if text == 'Iteration':
            text = 'Final'

        if self.widgets['results']['feature_window']:
            self.widgets['results']['feature_window'].destroy()

        self.widgets['results']['feature_window'] = FeatureWindow(self, self.analysis_results[text].events)
        self.flags['feature_win'] = False

    def create_tab(self, results, title):
        # Create tab and frames
        tab = ttk.Frame(self.widgets['results']['notebook'])
        header_separator = ttk.Separator(tab, style='primary.TSeparator')
        stats_frame = ttk.Frame(tab)
        event_frame = ttk.Frame(tab)
        footer_separator = ttk.Separator(tab, style='primary.TSeparator')
        feature_window_button = ttk.Button(
            tab,
            text="Feature Window",
            underline=0,
            command=self.create_feature_window,
            style='Small.secondary.Outline.TButton'
        )
        self.bind('<f>', lambda _: self.create_feature_window())
        tooltip.Hovertip(feature_window_button, 'F')

        # Tab layout
        tab.columnconfigure(0, weight=1)
        header_separator.grid(row=0, column=0, columnspan=2, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        stats_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(2, weight=5)
        event_frame.grid(row=2, column=0, sticky="nsew")
        feature_window_button.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=PADDING, pady=PADDING)
        footer_separator.grid(row=4, column=0, columnspan=2, sticky="nsew")

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
            values[-1].grid(row=row, column=1, sticky="e", padx=PADDING)
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

            row_width = max([len(stat[key]) for stat in table_stats]) + 2
            row_width *= self.fonts['small'].measure("0")
            heading_width = len(str(key)) + 2
            heading_width *= self.fonts['small_i'].measure("0")
            max_width[key] = row_width if row_width > heading_width else heading_width
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
        tree.tag_configure('outlier', background=self.colours['baseline'], foreground='white',
                           font=self.fonts['small'])
        tree.tag_configure('not_outlier', background='white', foreground=self.style.colors.secondary,
                           font=self.fonts['small'])

        # Set the headings
        for key in keys:
            if key not in self.table_columns:
                continue

            tree.heading(key, anchor='center', text=key, command=lambda _col=key:
                         self.sort_column(tree, _col, False))
            if key == 'ID':
                tree.column(key, width=max_width[key], anchor='w', stretch=1)
            else:
                tree.column(key, width=max_width[key], anchor='center', stretch=1)

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
        tree.bind("<BackSpace>", partial(self.outlier_event, title))

        # Scrollbar
        scrollbar = ttk.Scrollbar(event_frame, orient="vertical", command=tree.yview,
                                  style='secondary.Vertical.TScrollbar')
        tree.configure(yscrollcommand=scrollbar.set)

        # Table layout
        event_frame.rowconfigure(0, weight=1)
        scrollbar.grid(row=0, column=0, sticky="ns")
        event_frame.columnconfigure(1, weight=1)
        tree.grid(row=0, column=1, sticky="nsew")

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
        graph_frame.grid(row=1, column=0, sticky="nsew", pady=(0, PADDING))
        footer_separator = ttk.Separator(tab, style='primary.TSeparator')
        footer_separator.grid(row=2, column=0, sticky="nsew")

        # Graph frame
        if self.widgets['results']['notebook'].tabs():
            notebook_width = (self.widgets['results']['notebook'].winfo_reqwidth() - 4 * PADDING) / 100
        else:
            notebook_width = (self.width // 6 - 4 * PADDING) / 100
        length = len(trace_stats[0].keys()) + len(event_stats[0].keys())
        fig = Figure(layout='constrained', dpi=100, figsize=(notebook_width, 1.5 * length))
        # Trace stats
        x = np.arange(1, len(trace_stats) + 1)
        for i, key in enumerate(trace_stats[0].keys()):
            ax = fig.add_subplot(length, 1, i + 1)
            ax.axvline(x=len(trace_stats), color=self.colours['baseline'], linestyle='--', linewidth=0.5)
            ax.plot(x, [stats[key] for stats in trace_stats], label=key, color=self.colours['trace'],
                    linewidth=0.5, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
            ax.scatter(x, [stats[key] for stats in trace_stats], label=key, color=self.colours['threshold'],
                       path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
            ax.set_title(key, fontsize=SMALL_FONT)
            ax.tick_params(axis='both', which='both', labelsize=SMALL_FONT - 2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Event stats
        x = np.arange(1, len(event_stats) + 1)
        for i, key in enumerate(event_stats[0].keys()):
            ax = fig.add_subplot(length, 1, i + 1 + len(trace_stats[0].keys()))
            ax.axvline(x=len(trace_stats), color=self.colours['baseline'], linestyle='--', linewidth=0.5)
            ax.plot(x, [stats[key] for stats in event_stats], label=key, color=self.colours['trace'],
                    linewidth=0.5, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
            ax.scatter(x, [stats[key] for stats in event_stats], label=key, color=self.colours['threshold'],
                       path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
            ax.set_title(key, fontsize=SMALL_FONT)
            ax.tick_params(axis='both', which='both', labelsize=SMALL_FONT - 2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.supxlabel('Iteration', fontsize=SMALL_FONT)
        # Margins and share x ticks
        for axis in fig.get_axes():
            axis.margins(x=0.1, y=0.1)
            axis.format_coord = lambda x, y: f'y = {y:.4f}'
        for ax in fig.get_axes()[1:]:
            ax.sharex(fig.get_axes()[0])

        # Inner canvas and scrollbar
        inner_canvas = tk.Canvas(graph_frame)
        graph_scrollbar = ttk.Scrollbar(graph_frame, orient="vertical", style='secondary.Vertical.TScrollbar')
        # Configure the canvas size and scrollbar
        size = fig.get_size_inches() * fig.dpi
        inner_canvas.configure(width=size[0])
        inner_canvas.config(yscrollcommand=graph_scrollbar.set, scrollregion=(0, 0, 0, size[1]),
                            yscrollincrement=size[1] / (length * 10))
        graph_scrollbar.config(command=inner_canvas.yview)
        # Graph frame layout
        graph_frame.rowconfigure(2, weight=1)
        graph_scrollbar.grid(row=2, column=0, sticky="ns")
        graph_frame.columnconfigure(1, weight=1)
        inner_canvas.grid(row=2, column=1, sticky="nsew")

        # Inner frame inside inner canvas
        inner_frame = ttk.Frame(inner_canvas)
        inner_canvas.create_window(0, 0, window=inner_frame, anchor=tk.NW)
        # Plot inside inner frame... inside inner canvas
        canvas = FigureCanvasTkAgg(fig, inner_frame)
        # Create toolbar
        toolbar = NavigationToolbar2Tk(canvas, graph_frame, pack_toolbar=False)
        for child in toolbar.winfo_children():
            child.pack_forget()
            child.configure(background='white', border=1)
        for child in toolbar.winfo_children():
            if child != toolbar.children['!button4']:
                child.pack(side=tk.LEFT)
        toolbar.update()
        toolbar.push_current()
        toolbar.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=PADDING)
        toolbar_separator = ttk.Separator(graph_frame)
        toolbar_separator.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, PADDING))
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw_idle()
        canvas.flush_events()

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

        events = self.analysis_results[title].events
        for item in tree.selection():
            event_id = int(tree.item(item, 'text'))

            # Toggle the outlier flag
            if events.get(event_id)['Outlier']:
                events.get(event_id)['Outlier'] = False
                tree.item(item, tags='not_outlier')
            else:
                events.get(event_id)['Outlier'] = True
                tree.item(item, tags='outlier')

        event_coordinates = []
        for event in events.events:
            if not event['Outlier']:
                event_coordinates.append(event['Coordinates'])

        # Recalculate cutoff
        try:
            self.analysis_results[title].event_stats['Max Cutoff'] = np.round(
                np.percentile(
                    nn.calculate_cutoffs(
                        self.trace,
                        self.baseline,
                        self.initial_threshold,
                        np.asarray(event_coordinates, dtype=np.int64),
                        self.current_cutoff,
                        int(1 / self.time_step)
                    ),
                    25
                ),
                2
            )
        except IndexError:
            self.analysis_results[title].event_stats['Max Cutoff'] = self.analysis_options['estimate_cutoff'].get()

        self.analysis_options['cutoff'].set(self.analysis_results[title].event_stats['Max Cutoff'])
        self.flash_entry(self.widgets['analyse']['cutoff_input'])

        # Update the feature window figure
        if self.widgets['results']['feature_window']:
            self.widgets['results']['feature_window'].update_all_figures()

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
        extra = (event_coordinates[1] - event_coordinates[0]) * 10
        start = event_coordinates[0] - extra if event_coordinates[0] - extra > 0 else 0
        end = event_coordinates[1] + extra if event_coordinates[1] + extra < len(self.trace) else len(self.trace) - 1

        # Flash the event
        def remove():
            try:
                line[0].remove()
            except:
                return
            self.widgets['analysis_view']['fig'].canvas.draw_idle()
            self.widgets['analysis_view']['fig'].canvas.flush_events()

        ax = self.widgets['analysis_view']['fig'].axes[0]
        line = ax.plot(self.time_vector[event_coordinates[0]:event_coordinates[1] + 1],
                       self.trace[event_coordinates[0]:event_coordinates[1] + 1], 'r', linewidth=3, zorder=8)
        trace_min = np.min(self.trace[start:end])
        trace_max = np.max(self.trace[start:end])
        difference = 0.5 * abs(trace_max - trace_min)
        self.update_figure_xlimits((start * self.time_step + self.time_vector[0], end * self.time_step + self.time_vector[0]))
        ax.set(ylim=(trace_min - difference, trace_max + difference))
        self.widgets['analysis_view']['toolbar'].push_current()
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()
        self.after(500, remove)

    def pick_event(self, event):
        # Convert point to event coordinates
        point = np.round((event.artist.get_xdata()[event.ind[0]] - self.time_vector[0]) / self.time_step)

        # Get currently shown events
        if 'Final' in self.analysis_results:
            events = self.analysis_results['Final'].events
            label = 'Final'
        else:
            events = self.analysis_results['Estimate'].events
            label = 'Estimate'

        # Get the associated event
        event_id = None
        for event in events:
            coords = event['Coordinates']
            if coords[0] <= (point + 5) and coords[1] >= (point - 5):
                event_id = event['ID']
                break

        if event_id is None:
            return

        # Select the tab
        for i in self.widgets['results']['notebook'].tabs():
            if self.widgets['results']['notebook'].tab(i, "text") == label:
                self.widgets['results']['notebook'].select(i)
                break

        # Get the treeview
        tab = self.widgets['results']['notebook'].nametowidget(self.widgets['results']['notebook'].select())
        tree = tab.winfo_children()[2].winfo_children()[0]

        # Select the event
        for item in tree.get_children():
            if tree.item(item, 'text') == str(event_id):
                tree.selection_set(item)
                tree.focus(item)
                tree.see(item)
                dummy = tk.Event()
                dummy.widget = tree
                self.jump_to_event(label, dummy)
                break

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

    def flash_entry(self, entry):
        current = entry.cget('style')

        self.style.configure(f'flash.{current}', bordercolor=self.style.colors.primary,
                             relief='raised', foreground=self.style.colors.primary)
        for i in np.arange(0, 10, 2):
            self.after(i * 250, lambda: entry.configure(style=f'flash.{current}'))
            self.after((i + 1) * 250, lambda: entry.configure(style=current))
        self.after(3000, lambda: entry.configure(style=current))

    def downsample_line(self, x, y, xlim, max_points = None):
        if max_points is None:
            max_points = int(self.widgets['analysis_view']['canvas'].get_width_height()[0])
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x_visible = x[mask]
        y_visible = y[mask]
        if len(x_visible) <= max_points:
            return x_visible, y_visible

        idx_bins = np.array_split(np.arange(len(x_visible)), max_points)
        x_bins = [x_visible[idx] for idx in idx_bins]
        y_bins = [y_visible[idx] for idx in idx_bins]

        x_out = np.empty(2 * len(x_bins))
        y_out = np.empty(2 * len(y_bins))
        for i, (xb, yb) in enumerate(zip(x_bins, y_bins)):
            x_out[2 * i] = xb[0]
            x_out[2 * i + 1] = xb[-1]
            y_out[2 * i] = yb.min()
            y_out[2 * i + 1] = yb.max()

        return x_out, y_out

    def downsample_events(self, xlim):
        # Total width budget
        width = int(self.widgets['analysis_view']['canvas'].get_width_height()[0])
        # Select only events that overlap xlim
        visible = []
        for start, end in self.event_coordinates:
            if self.time_vector[end] >= xlim[0] and self.time_vector[start] <= xlim[1]:
                visible.append((start, end))
        if len(visible) == 0:
            return np.array([], dtype=object), np.array([], dtype=object)

        # At least 2 pixels per segment
        per_seg = max(2, width // len(visible))

        x_out, y_out = [], []
        for start, end in self.event_coordinates:
            # Grab and clamp each segment
            seg_x = self.time_vector[start:end + 1]
            seg_y = self.trace[start:end + 1]
            mask = (seg_x >= xlim[0]) & (seg_x <= xlim[1])
            if not mask.any():
                continue
            seg_x = seg_x[mask]
            seg_y = seg_y[mask]

            # Downsample this segment with its own budget
            xs, ys = self.downsample_line(
                seg_x, seg_y,
                xlim=(seg_x[0], seg_x[-1]),
                max_points=per_seg
            )

            x_out.extend(xs)
            x_out.append(None)
            y_out.extend(ys)
            y_out.append(None)

        return np.asarray(x_out, dtype=object), np.asarray(y_out, dtype=object)

    def downsample_all(self):
        ax = self.widgets['analysis_view']['fig'].axes[0]
        xlim = ax.get_xlim()
        thresh = 0
        for line in ax.get_lines():
            if line.get_label() not in self.colours.keys():
                continue
            if line.get_label() == 'threshold':
                if thresh == 0:
                    line.set_data(*self.downsample_line(self.time_vector, self.baseline + self.threshold, xlim))
                else:
                    line.set_data(*self.downsample_line(self.time_vector, self.baseline - self.threshold, xlim))
                thresh += 1
            elif line.get_label() == 'events':
                line.set_data(*self.downsample_events(xlim))
            else:
                line.set_data(*self.downsample_line(self.time_vector, getattr(self, line.get_label()), xlim))

    def toolbar_key_press(self, figure_id, event):
        key_press_handler(event, self.widgets[figure_id]['canvas'], self.widgets[figure_id]['toolbar'])

    def toolbar_wrapper(self, method):
        def wrapper(*args, **kwargs):
            method(*args, **kwargs)
            ax = self.widgets['analysis_view']['fig'].axes[0]
            # Downsample
            self.downsample_all()
            # Redraw
            ax.figure.canvas.draw_idle()
            ax.figure.canvas.flush_events()
        return wrapper

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
        self.initial_threshold = self.analysis_results[label].initial_threshold
        self.calculation_trace = self.analysis_results[label].calculation_trace
        self.event_coordinates = self.analysis_results[label].event_coordinates
        self.current_cutoff = self.analysis_options['estimate_cutoff'].get() if label == 'Estimate' else (
                              self.analysis_options['cutoff'].get())

        # Extract features
        events = Events(label)
        events.events = nn.simple_extractor(self.trace, self.baseline, self.event_coordinates, int(1 / self.time_step))
        # Store maximum cutoff
        max_cutoffs = nn.calculate_cutoffs(self.trace, self.baseline, self.initial_threshold, self.event_coordinates,
                                           self.current_cutoff, int(1 / self.time_step))
        events.add_feature('Max Cutoff', max_cutoffs)
        self.analysis_results[label].events = events
        try:
            self.analysis_results[label].event_stats['Max Cutoff'] = np.percentile(max_cutoffs, 25)
        except IndexError:
            self.analysis_results[label].event_stats['Max Cutoff'] = self.current_cutoff

        # Update the results window
        self.update_results()
        self.update()
        # Show the vectors
        self.update_figure(title=label)

        # Update the direction for estimate only
        if label == 'Estimate':
            self.analysis_options['event_direction'].set(self.analysis_results[label].event_direction)
            self.flash_entry(self.widgets['analyse']['event_direction_input'])

        # Update the cutoff
        self.analysis_options['cutoff'].set(np.round(self.analysis_results[label].event_stats['Max Cutoff'], 2))
        self.flash_entry(self.widgets['analyse']['cutoff_input'])

    def load_trace(self):
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        path = filedialog.askopenfilename(
            initialdir="/" if self.trace_information['path'].get() == ''
                           else os.path.split(self.trace_information['path'].get())[0],
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
                with open(path, newline='') as csvfile:
                    dialect = csv.Sniffer().sniff(csvfile.read(1024))
                    csvfile.seek(0)
                    reader = csv.reader(csvfile, dialect)
                    next(reader)
                    trace = []
                    time_vector = []
                    for row in reader:
                        trace.append(row[1])
                        time_vector.append(row[0])
                    trace = np.asarray(trace, dtype=float)
                    time_vector = np.asarray(time_vector, dtype=float)

            else:
                raise ValueError('File type not supported')
            time_step = time_vector[1] - time_vector[0]
        except Exception as e:
            messagebox.showerror('Error', f'Loading failed: {repr(e)}')
            return
        print(f'Loading time: {time.time() - current_time:.4f}')

        self.reset_all()
        # Update the trace vectors
        self.trace = trace
        self.time_vector = time_vector
        self.time_step = time_step
        # Show the trace information and trace vector
        self.update_load(path)
        # Initialise the slider
        self.init_slider()
        # Update the figure
        self.update_figure(title='Trace', retain_view=False)
        # Set the z-score
        self.analysis_options['z_score'].set(np.round(norm.ppf(1 - ((1 / len(self.trace)) / 2)), 3))
        self.default_analysis_options['z_score'] = self.analysis_options['z_score'].get()
        self.flash_entry(self.widgets['analyse']['z_score_input'])
        # Disable iterate options
        self.toggle_iterate_options(['disabled'])
        # Set the flag
        self.flags['loaded'] = True

    def landing_load(self):
        self.load_trace()

        if self.flags['loaded']:
            self.toggle_landing_page('hide')

    def check_thread(self, thread, results, errors, current_time, title):
        self.flags['running'] = True
        if thread.is_alive():
            if errors.empty():
                self.after(100, lambda: self.check_thread(thread, results, errors, current_time, title))
            else:
                self.widgets['analysis_view']['progress_bar'].stop()
                self.flags['running'] = False
                messagebox.showerror('Error', f'{title} failed: {repr(errors.get())}')
            return
        thread.join()
        self.update_idletasks()
        self.flags['running'] = False
        results = results.get()
        self.widgets['analysis_view']['progress_bar'].stop()
        print(f'{title} time: {time.time() - current_time:.4f}')

        # Update analysis results
        self.process_results(results)
        # Set the focus and flag
        if title == 'Estimate':
            self.widgets['analyse']['iterate'].focus_set()
            self.flags['estimated'] = True
            # Enable iterate options
            self.toggle_iterate_options(['!disabled'])
        else:
            self.widgets['results']['save'].focus_set()
            self.flags['iterated'] = True

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
                self.analysis_options['replace_factor'].get(),
                self.analysis_options['replace_gap'].get(),
                self.analysis_options['threshold_window'].get(),
                self.analysis_options['z_score'].get(),
            )
        )
        thread.start()
        self.check_thread(thread, results, errors, current_time, 'Estimate')

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
        self.check_thread(thread, results, errors, current_time, 'Iteration')

    def save_results(self):
        if not self.flags['iterated'] and not self.flags['estimated']:
            messagebox.showerror('Error', 'No results to save')
            return
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return
        if self.flags['iterated']:
            results = copy.deepcopy(self.analysis_results['Final'].events)
            label = 'final'
        else:
            results = copy.deepcopy(self.analysis_results['Estimate'].events)
            label = 'estimate'
        if not results:
            messagebox.showerror('Error', 'No results to save')
            return

        # Save the results
        path = tk.filedialog.asksaveasfile(
            initialfile=basename(self.trace_information['path'].get()).split('.')[0] + f'_{label}.csv',
            defaultextension=".csv",
            initialdir=os.path.split(self.trace_information['path'].get())[0],
            title="Choose a location",
            filetypes=([("CSV files", "*.csv")])
        )
        if path == '' or path is None:
            return

        # If you don't do this, DictWriter will ruin your life :( (** 2 bcos of excel cell size limit)
        units = self.trace_information['units'].get()
        amp_key = 'Amplitude (A)' if units == 'A' else f'Amplitude ({units[0]}A)'
        area_key = 'Area (C)' if units == 'A' else f'Area ({units[0]}C)'
        for event in results:
            packed = b"".join(struct.pack("d", num) for num in list(event.pop('Vector')))
            event['Vector (Base64)'] = base64.b64encode(packed).decode()
            event['Duration (ms)'] = event.pop('Duration')
            event['Max Cutoff (Hz)'] = event.pop('Max Cutoff')
            event[area_key] = event.pop('Area')
            event[amp_key] = event.pop('Amplitude')

        fieldnames = [
            'ID',
            'Direction',
            amp_key,
            'Duration (ms)',
            area_key,
            'Max Cutoff (Hz)',
            'Coordinates',
            'Outlier',
            'Vector (Base64)',
        ]

        with open(path.name, 'w', encoding='utf8', newline='') as path:
            w = csv.DictWriter(path, fieldnames)
            w.writeheader()
            w.writerows(results.events)

        # Success
        messagebox.showinfo('Success', f'Saved {label} results')

        # Set the flag
        self.flags['saved'] = True
