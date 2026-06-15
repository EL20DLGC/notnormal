# Copyright (C) 2025-2026 Dylan Charnock <el20dlgc@leeds.ac.uk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module is the GUI for the NotNormal algorithm. It is inefficient, has bugs, but it works and is functional.
I was told we needed a GUI, so I wrote the first version very quickly, while relatively new to Python. It was spaghetti.
Over the time I have been asked if things can be added, I have, and it has only got worse. I will never de-spag this,
I don't like to change it. I am working on a complete rewrite that has become incredibly ambitious, maybe you will see
that one day.
"""

import csv
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
from functools import partial, lru_cache
from webbrowser import open_new
from idlelib import tooltip
from os.path import basename
from threading import Thread
from tkinter import ttk, colorchooser, messagebox, filedialog
from tkinter.font import Font
from typing import Optional
import numpy as np
import pyabf
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib import patheffects as pe, rc, style as mplstyle
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from ttkbootstrap import Style, Floodgauge
from notnormal import extract as nn
from notnormal.models.base import IterateResults

_COMPILED = cython.compiled
_PADDING = 6
_WINDOW_PADDING = 3
_ENTRY_WIDTH = 9
_LARGE_FONT = 16
_MEDIUM_FONT = 14
_SMALL_FONT = 10


@lru_cache(maxsize=200)
def _get_edges(length, max_points):
    # Equally spaced bin edges
    k = np.arange(max_points, dtype=np.int64)
    starts = (k * np.int64(length)) // np.int64(max_points)
    ends = np.empty_like(starts)
    ends[:len(ends) - 1] = starts[1:]
    ends[len(ends) - 1] = length
    return starts, ends


class CustomFigureCanvas(FigureCanvasTkAgg):
    def __init__(self, figure, window, root, resize_callback = None):
        super().__init__(figure, window)
        self.root = root
        self.resize_callback = resize_callback
        self.after_id = None
        self.last_event = None

    def resize(self, event = None, instant = False):
        # Create an event if needed
        if event is None:
            self.root.update_idletasks()
            event = tk.Event()
            event.width = self.get_tk_widget().winfo_width()
            event.height = self.get_tk_widget().winfo_height()

        # No real event or same as last
        if event.width <= 1 or event.height <= 1:
            return
        if (self.last_event is not None and (event.width, event.height) == (self.last_event.width, self.last_event.height)
            and not instant):
            return
        self.last_event = event

        # Coalesce
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)

        # Instant fire or queue
        if instant:
            self.do_resize(instant=True)
            self.draw()
            self.flush_events()
        else:
            self.after_id = self.root.after(250, self.queue_resize)

    def queue_resize(self):
        self.after_id = None
        self.root.after_idle(self.do_resize)

    def do_resize(self, instant = False):
        # Do the resize
        event = self.last_event
        self.last_event = None
        if event is None:
            return
        super(CustomFigureCanvas, self).resize(event)

        # Callback post-resize
        if self.resize_callback is not None:
            if instant:
                self.resize_callback()
            else:
                self.root.after_idle(self.resize_callback)


class FeatureWindow(tk.Toplevel):
    def __init__(self, master, events):
        super().__init__(master)

        self.hide()
        self.master = master
        self.events = events
        # Destroy protocol
        self.protocol('WM_DELETE_WINDOW', self.hide)
        self.master.bind('<Destroy>', lambda e: self.destroy() if e.widget is self.master else None, add='+')
        # Always on top of main
        self.transient(master)
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
        self.height = int(self.master.screen_height * 0.4)
        self.width = int(self.master.screen_width * 0.4)
        self.geometry(f'{self.width}x{self.height}+{int(self.master.screen_width / 2 - self.width / 2)}+{int(self.master.screen_height / 2 - self.height / 2)}')
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
                    "x": {"label": "Feature", "relim": True, "type": "feature", "values": self.features, "default": self.features[4]},
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
                    "x": {"label": "Feature 1", "relim": True, "type": "feature", "values": self.features, "default": self.features[2]},
                    "y": {"label": "Feature 2", "relim": True, "type": "feature", "values": self.features, "default": self.features[3]},
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
                    "xs": {"label": "Feature 1", "relim": True, "type": "feature", "values": self.features, "default": self.features[2]},
                    "ys": {"label": "Feature 1", "relim": True, "type": "feature", "values": self.features, "default": self.features[3]},
                    "zs": {"label": "Feature 3", "relim": True, "type": "feature", "values": self.features, "default": self.features[4]},
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
                "label": "Event Plot",
                "function": "plot",
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
                    "linewidth": {"label": "Linewidth", "relim": False, "type": "float", "values": [0, 5, 0.5],
                                   "default": 1},
                    "alpha": {"label": "Alpha", "relim": False, "type": "float", "values": [0, 1, 0.05], "default": 1},
                    "color": {"label": "Colour", "relim": False, "type": "colour", "default": master.colours["trace"]},
                },
                "feature_filters": {
                    "Outlier": {"label": "Show Outliers", "relim": False, "type": "bool", "onvalue": "all",
                                "default": False},
                    "Direction": {"label": "Direction", "relim": False, "type": "str", "values": ["up", "down", "all"],
                                  "default": "all"},
                }
            },
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
        self.update_all_figures(relim=True)

        # Select the first tab
        self.set_current_tab(self.figure_configs[0]['label'])
        self.show()

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
        self.windows['left'].columnconfigure(0, weight=1)
        self.windows['left'].rowconfigure(0, weight=1)
        self.windows['options'].grid(row=0, column=0, sticky="nsew", padx=(_WINDOW_PADDING, 0), pady=_WINDOW_PADDING)
        # Right layout
        self.windows['right'].rowconfigure(0, weight=1)
        self.windows['right'].columnconfigure(0, weight=1)
        self.windows['feature_view'].grid(row=0, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

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
        self.widgets['options']['title'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0))
        self.windows['options_internal'].grid(row=1, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

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
        self.widgets['feature_view']['title'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0))
        self.windows['feature_view_internal'].grid(row=1, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

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
                widgets[opt + '_label'].grid(row=i, column=0, sticky="nsew", padx=_PADDING)

                # Entry
                if options[opt]['type'] in ['str', 'feature']:
                    widgets[opt] = ttk.Combobox(
                        widgets['frame'],
                        textvariable=tk_var[opt],
                        width=_ENTRY_WIDTH,
                        justify='center',
                        values=options[opt]['values'],
                        state='readonly'
                    )
                    widgets[opt].bind("<<ComboboxSelected>>", lambda _, x = options[opt]['relim']: (
                                                                self.update_figure(figure_config, relim=x)))
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=_PADDING)
                    widgets[opt].configure(font=self.master.fonts['small'])
                elif options[opt]['type'] in ['int', 'float']:
                    widgets[opt] = ttk.Spinbox(
                        widgets['frame'],
                        textvariable=tk_var[opt],
                        width=_ENTRY_WIDTH,
                        justify='center',
                        from_=options[opt]['values'][0],
                        to=options[opt]['values'][1],
                        increment=options[opt]['values'][2],
                        command=partial(self.update_figure, figure_config, relim=options[opt]['relim'])
                    )
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=_PADDING)
                    widgets[opt].configure(font=self.master.fonts['small'])
                elif options[opt]['type'] == 'bool':
                    widgets[opt] = ttk.Checkbutton(
                        widgets['frame'],
                        variable=tk_var[opt],
                        width=_ENTRY_WIDTH - 5,
                        onvalue=True,
                        offvalue=False,
                        command=partial(self.update_figure, figure_config, relim=options[opt]['relim']),
                        style='Roundtoggle.Toolbutton'
                    )
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=_PADDING)
                elif options[opt]['type'] == 'colour':
                    widgets[opt] = ttk.Button(
                        widgets['frame'],
                        command=partial(self.update_colour, figure_config, opt),
                        width=_ENTRY_WIDTH + 2
                    )
                    self.master.style.configure(
                        f'{label.replace(" ", "")}_{opt}_colour.TButton', background=tk_var[opt].get(), borderwidth=0,
                        font=self.master.fonts['small_b'])
                    self.master.style.map(f'{label.replace(" ", "")}_{opt}_colour.TButton',
                                          background=[('active', tk_var[opt].get())])
                    widgets[opt].configure(style=f'{label.replace(" ", "")}_{opt}_colour.TButton')
                    widgets[opt].grid(row=i, column=1, sticky="e", padx=_PADDING, pady=6)

                # Separator
                if i < (2 * opt_length) - 1:
                    widgets[opt + '_separator'] = ttk.Separator(widgets['frame'], orient='horizontal')
                    widgets[opt + '_separator'].grid(row=i + 1, column=0, columnspan=2, sticky="nsew")

                i += 2

        # Add config button
        widgets['config_button'] = ttk.Button(widgets['frame'], text="Edit Configuration", underline=0,
                                              command=partial(self.create_edit_window, figure_config),
                                              style='secondary.Outline.TButton')
        widgets['config_button'].grid(row=i, column=0, columnspan=2, sticky="nsew", pady=(0, _PADDING), padx=_PADDING)

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
        widgets['canvas'] = CustomFigureCanvas(widgets['fig'], widgets['frame'], self)
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
        widgets['toolbar'].grid(row=0, column=0, sticky='nsew', padx=_PADDING)
        widgets['toolbar_separator'].grid(row=1, column=0, sticky="nsew")
        widgets['canvas'].get_tk_widget().grid(row=2, column=0, sticky='nsew', padx=2 * _PADDING, pady=2 * _PADDING)

        # Feature view
        widgets['fig'].add_subplot(111, **sub_plot_init)

        # Bind events
        widgets['canvas'].mpl_connect("pick_event", self.pick_event)

    def update_events(self, events):
        self.events = events
        self.update_all_figures(relim=True)

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

        try:
            ax.clear()

            # Get the filtered event IDs
            event_ids = self.get_filtered_ids(figure_config)

            # Get figure options
            options = dict()
            current_features = []
            pick_features = []
            for option in plot_options.keys():
                if plot_options[option]['type'] == 'feature':
                    raw_feature = self.plot_options[label][option].get()
                    options[option] = self.events.get_feature(raw_feature, event_ids)

                    feature = raw_feature
                    units = self.master.trace_information['units'].get()

                    # Bit of spag bol for units
                    if feature == 'Duration':
                        feature = 'Duration ($ms$)'
                    elif feature == 'Area':
                        feature = 'Area ($C$)' if units == 'A' else f'Area (${units[0]}C$)'
                    elif feature == 'Amplitude':
                        feature = 'Amplitude ($A$)' if units == 'A' else f'Amplitude (${units[0]}A$)'
                    elif feature == 'Max Cutoff':
                        feature = 'Max Cutoff ($Hz$)'

                    pick_features.append(raw_feature)
                    current_features.append(feature)
                else:
                    options[option] = self.plot_options[label][option].get()

            # Event plot is a special case
            if label == 'Event Plot':
                vecs = self.events.get_feature('Vector', event_ids)
                x_parts = []
                y_parts = []
                for y in vecs:
                    y = np.asarray(y, dtype=float)
                    x = np.arange(len(y), dtype=float)

                    x_parts.append(x)
                    y_parts.append(y)

                    # Break between events
                    x_parts.append(np.array([np.nan]))
                    y_parts.append(np.array([np.nan]))

                if x_parts:
                    x_all = np.concatenate(x_parts)
                    y_all = np.concatenate(y_parts)
                else:
                    x_all = np.array([])
                    y_all = np.array([])

                options['_series'] = [x_all, y_all]
                units = self.master.trace_information['units'].get()
                current_features = ['Sample', 'Current ($A$)' if units == 'A' else f'Current (${units[0]}A$)']
                pick_features.append('Vector')

            # Plot
            plot = getattr(ax, function)(*options.pop('_series', []), **options, **plot_init, picker=True)

            # Attach event ids if possible
            if isinstance(plot, (PathCollection, Path3DCollection)):
                plot._event_ids = np.asarray(event_ids, dtype=object)

            # Get axis options
            options = dict()
            for option in axis_options.keys():
                options[option] = self.axis_options[label][option].get()

            # Set axis options
            ax.set(**options, **axis_init, label=str(pick_features))

            # Title
            if 'title' not in axis_init.keys():
                ax.set_title(self.events.label, fontsize=_LARGE_FONT)
            # Labels
            if 'xlabel' not in axis_init.keys():
                ax.set_xlabel(current_features[0], fontsize=_MEDIUM_FONT)
            if 'ylabel' not in axis_init.keys() and len(current_features) > 1:
                ax.set_ylabel(current_features[1], fontsize=_MEDIUM_FONT)
            if 'zlabel' not in axis_init.keys() and len(current_features) > 2:
                ax.set_zlabel(current_features[2], rotation=90, fontsize=_MEDIUM_FONT)

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

        except Exception:
            ax.set_xlim(view_limits[0])
            ax.set_ylim(view_limits[1])
            return

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
            f'{self.width}x{self.height}+{int(self.master.screen_width / 2 - self.width / 2)}+{int(self.master.screen_height / 2 - self.height / 2)}')
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
        windows['config'].grid(row=0, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

        # Title and inner frame
        widgets['title'] = ttk.Label(windows['config'], text="JSON Configuration", style='primary.Inverse.TLabel',
                                     anchor='center')
        windows['config_internal'] = ttk.Frame(windows['config'])
        # Layout
        windows['config'].columnconfigure(0, weight=1)
        windows['config'].rowconfigure(1, weight=1)
        widgets['title'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0))
        windows['config_internal'].grid(row=1, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

        # Entry
        size = self.master.fonts['small'].measure('    ')
        widgets['entry'] = tk.Text(windows['config_internal'], tabs=size, font=self.master.fonts['small'])
        windows['config_internal'].columnconfigure(0, weight=1, uniform=1)
        windows['config_internal'].rowconfigure(0, weight=1, uniform=1)
        widgets['entry'].grid(row=0, column=0, sticky="nsew", padx=_PADDING, pady=(0, _PADDING))

        widgets['add_button'] = ttk.Button(windows['config_internal'], text="Confirm", underline=0,
                                           command=partial(self.edit_figure_config, figure_config),
                                           style='secondary.Outline.TButton')
        widgets['add_button'].grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, _PADDING), padx=_PADDING)

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

    def remove_figure_config(self, figure_config: dict):
        # Remove the figure frame
        self.widgets['feature_view'][figure_config['label']]['frame'].destroy()
        # Remove the options tab
        self.widgets['options'][figure_config['label']]['frame'].destroy()
        # Update the current figure
        self.current_figure = None
        self.set_current_figure()

    def hide(self):
        self.attributes('-alpha', 0)

    def show(self):
        self.attributes('-alpha', 0)
        self.deiconify()
        self.update()
        self.widgets['feature_view'][self.current_figure]['canvas'].resize(instant=True)
        self.update()
        self.attributes('-alpha', 1)
        self.lift(self.master)

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
        self.widgets['feature_view'][self.current_figure]['canvas'].resize(instant=True)

    def pick_event(self, event):
        # Ignore artists that do not support picking
        if not hasattr(event.artist, '_event_ids'):
            return
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return

        # Get all
        event_ids = event.artist._event_ids
        inds = np.asarray(event.ind, dtype=int)
        inds = inds[(inds >= 0) & (inds < len(event_ids))]
        if len(inds) == 0:
            return

        # Pick the nearest if several points overlap
        try:
            if isinstance(event.artist, Path3DCollection):
                xs, ys, _ = event.artist._offsets3d
                points = np.column_stack((np.asarray(xs)[inds], np.asarray(ys)[inds]))
            elif isinstance(event.artist, PathCollection):
                offsets = event.artist.get_offsets()
                points = np.asarray(offsets)[inds, :2]
            else:
                return

            # Convert data coordinates and find the closest point
            display_points = event.artist.axes.transData.transform(points)
            mouse = np.array([event.mouseevent.x, event.mouseevent.y])
            ind = inds[np.argmin(np.sum((display_points - mouse) ** 2, axis=1))]
        except Exception:
            ind = inds[0]

        event_id = event_ids[int(ind)]

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
                tree.focus_set()
                tree.see(item)
                dummy = tk.Event()
                dummy.widget = tree
                self.master.jump_to_event(label, dummy)
                break


class NotNormalGUI(tk.Tk):
    """
    Init
    """

    def __init__(self):
        super().__init__()

        self.hide()
        # Loose focus pls
        self.bind("<FocusOut>", lambda event: self.wm_attributes('-topmost', 0))
        self.attributes('-topmost', True)
        self.after_idle(lambda: self.attributes('-topmost', False))
        # Title
        self.title("Not Normal")
        self.tk.call('tk', 'appname', 'NotNormal')
        # Background
        self.configure(bg="black")
        # Icon
        if hasattr(sys, '_MEIPASS'):
            self.data_path = os.path.join(sys._MEIPASS, 'data')
        else:
            self.data_path = os.path.join(__file__, '..', '..', 'data')
        if sys.platform == 'win32':
            self.icon = os.path.join(self.data_path, 'logo.ico')
            self.iconbitmap(True, self.icon)
        else:
            self.icon = tk.PhotoImage(file=os.path.join(self.data_path, 'logo.png'))
            self.iconphoto(True, self.icon)
        # Root window size and resizability
        self.screen_height = self.winfo_screenheight()
        self.height = int(self.screen_height * 0.8)
        self.landing_height = 400
        self.screen_width = self.winfo_screenwidth()
        self.width = int(self.screen_width * 0.8)
        self.landing_width = 700
        self.default_geometry = f'{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}'
        self.landing_geometry = f'{self.landing_width}x{self.landing_height}+{int(self.screen_width / 2 - self.landing_width / 2)}+{int(self.screen_height / 2 - self.landing_height / 2)}'
        self.geometry(self.landing_geometry)
        self.resizable(True, True)
        # DPI scaling
        self.tk.call('tk', 'scaling', self.winfo_fpixels('1i') / 72)
        # Dictionary to store all widgets
        self.widgets = {}
        self.flashing = {}
        self.previous_pane_sizes = {}
        # Dictionary to store all windows
        self.windows = {}
        # Figure options
        self.figure_options = {}
        self.default_figure_options = {}
        # Analysis options
        self.analysis_options = {}
        self.default_analysis_options = {}
        # Temporary analysis options
        self.analysis_options['parallel'] = tk.BooleanVar(value=True)
        self.default_analysis_options['parallel'] = self.analysis_options['parallel'].get()
        # Analysis results
        self.analysis_results = {}
        self.last_called = None
        self.table_columns = ['ID', 'Area', 'Max Cutoff', 'Duration', 'Amplitude']
        # Trace information variables
        self.trace_information = {}
        self.trace = None
        self.time_vector = None
        self.time_step = None
        # Range variables
        self.current_range = None
        self.pending_range = None
        self.dragged_range = None
        self.range_line = None
        # Flags for ordering
        self.flags = dict(loaded=False, estimated=False, iterated=False, saved=False, running=False, feature_win=False)
        self.after_id = None

        # Initialise the style
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
        self.toggle_landing_page(True)

    def init_style(self):
        # Base style
        self.style = Style(theme='pulse')
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
            'small': Font(family=family, name='small', size=_SMALL_FONT, weight='normal', slant='roman'),
            'small_i': Font(family=family, name='small_i', size=_SMALL_FONT, weight='normal', slant='italic'),
            'small_b': Font(family=family, name='small_b', size=_SMALL_FONT, weight='bold', slant='roman'),
            'small_ib': Font(family=family, name='small_ib', size=_SMALL_FONT, weight='bold', slant='italic'),
            'medium': Font(family=family, name='medium', size=_MEDIUM_FONT, weight='normal', slant='roman'),
            'medium_i': Font(family=family, name='medium_i', size=_MEDIUM_FONT, weight='normal', slant='italic'),
            'medium_b': Font(family=family, name='medium_b', size=_MEDIUM_FONT, weight='bold', slant='roman'),
            'large': Font(family=family, name='large', size=_LARGE_FONT, weight='normal', slant='roman'),
            'large_i': Font(family=family, name='large_i', size=_LARGE_FONT, weight='normal', slant='italic'),
            'large_b': Font(family=family, name='large_b', size=_LARGE_FONT, weight='bold', slant='roman')
        }
        # Initialise the style
        self.style.configure('.', font=self.fonts['small'])
        # Labels
        self.style.configure('TLabel', anchor=tk.W, font=self.fonts['small'])
        self.style.configure('primary.Inverse.TLabel', anchor=tk.CENTER, justify=tk.CENTER, font=self.fonts['large'])
        self.style.configure('secondary.TLabelframe.Label', anchor=tk.CENTER, justify=tk.CENTER,
                             foreground=self.style.colors.secondary, font=self.fonts['medium'])
        # Buttons
        self.style.configure('TButton', anchor=tk.CENTER, padding=_PADDING, font=self.fonts['medium_i'])
        self.style.configure('primary.TButton', anchor=tk.CENTER, padding=_PADDING, font=self.fonts['large_i'])
        self.style.configure('secondary.Outline.TButton', anchor=tk.CENTER, padding=_PADDING,
                             font=self.fonts['medium_i'])
        self.style.configure('Small.secondary.Outline.TButton', anchor=tk.CENTER, padding=_PADDING,
                             font=self.fonts['small_i'])
        # Entries
        self.style.configure('TCombobox', padding=_PADDING, font=self.fonts['small_b'])
        self.style.map('TCombobox', fieldbackground=[('readonly', 'white')],  background=[('readonly', 'white')])
        layout = self.style.layout('TCombobox')
        layout[0][1]['children'][0] = ('TSpinbox.downarrow', {'side': 'right', 'sticky': 'e'})
        self.style.layout('TCombobox', layout)
        self.style.configure('TSpinbox', padding=_PADDING, font=self.fonts['small_b'])
        self.style.configure('TEntry', padding=_PADDING, font=self.fonts['small_b'])
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
        # Progress bar
        self.style.configure('Horizontal.TFloodgauge', thickness=30, barsize=60)

        # Matplotlib
        mplstyle.use('fast')
        rc('text', usetex=False, hinting='force_autohint', hinting_factor=8)
        rc('mathtext', fontset='dejavusans', default='it')
        rc('font', size=_SMALL_FONT, family='sans-serif', **{'sans-serif': ['DejaVu Sans']})
        rc('figure', dpi=float(self.winfo_fpixels('1i')))
        rc('grid', color='0.8', linewidth=0.5)
        rc('path', simplify=True, simplify_threshold=0.111111111111)
        rc('agg.path', chunksize=200000)
        rc('lines', antialiased=True)

    def init_layout(self):
        # Main window
        self.windows['main'] = ttk.Frame(self)
        self.windows['main'].pack(fill=tk.BOTH, expand=True)

        # Main pane
        self.windows['paned'] = ttk.PanedWindow(self.windows['main'], orient=tk.HORIZONTAL, style='secondary.Horizontal.TPanedwindow')
        self.style._build_configure("Sash", gripcount=10, sashthickness=_PADDING)
        self.windows['main'].rowconfigure(0, weight=1)
        self.windows['main'].columnconfigure(0, weight=1)
        self.windows['paned'].grid(row=0, column=0, sticky="nsew")

        # Left window
        self.windows['left'] = ttk.Frame(self.windows['paned'], style='secondary.TFrame', width=self.width // 6)
        self.windows['left'].grid_propagate(False)
        # Center window
        self.windows['center'] = ttk.Frame(self.windows['paned'], style='secondary.TFrame')
        # Right window
        self.windows['right'] = ttk.Frame(self.windows['paned'], style='secondary.TFrame', width=self.width // 6)
        self.windows['right'].grid_propagate(False)

        # Defer resizing till after layout
        def set_default_pane_widths():
            self.update_idletasks()
            self.windows['left'].configure(width=max(self.width // 6, int(1.1 * self.windows['left'].winfo_reqwidth())))
            self.windows['right'].configure(width=max(self.width // 6, int(1.1 * self.windows['right'].winfo_reqwidth())))
        self.after_idle(set_default_pane_widths)

        # Sash events
        self.windows['paned'].bind('<Double-1>', self.double_click_sash, add='+')

        # Main layout
        self.windows['paned'].add(self.windows['left'], weight=0)
        self.windows['paned'].add(self.windows['center'], weight=1)
        self.windows['paned'].add(self.windows['right'], weight=0)

        # Trace loading and information window
        self.windows['load'] = ttk.Frame(self.windows['left'], style='primary.TFrame')
        # Analyse window
        self.windows['analyse'] = ttk.Frame(self.windows['left'], style='primary.TFrame')
        # Analysis view window
        self.windows['analysis_view'] = ttk.Frame(self.windows['center'], style='primary.TFrame')
        # Results window
        self.windows['results'] = ttk.Frame(self.windows['right'], style='primary.TFrame')

        # Left layout
        self.windows['left'].columnconfigure(0, weight=1)
        self.windows['left'].rowconfigure(0, weight=1)
        self.windows['load'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0), padx=(_WINDOW_PADDING, 0))
        self.windows['left'].rowconfigure(1, weight=2)
        self.windows['analyse'].grid(row=1, column=0, sticky="nsew", pady=_WINDOW_PADDING, padx=(_WINDOW_PADDING, 0))
        # Center layout
        self.windows['center'].columnconfigure(0, weight=1)
        self.windows['center'].rowconfigure(0, weight=1)
        self.windows['analysis_view'].grid(row=0, column=0, sticky="nsew", pady=_WINDOW_PADDING, padx=0)
        # Right layout
        self.windows['right'].columnconfigure(0, weight=1)
        self.windows['right'].rowconfigure(0, weight=1)
        self.windows['results'].grid(row=0, column=0, sticky="nsew", pady=_WINDOW_PADDING, padx=(0, _WINDOW_PADDING))

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
                         "Median filtering window size for bounding events (estimate and iterate)", hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['estimate_cutoff_input'],
                         'Cutoff frequency (estimate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['threshold_window_input'],
                         'Window size for threshold calculation (estimate and iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['z_score_input'],
                         'Z-score for threshold calculation (estimate and iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['features_input'],
                         'Feature extraction type (estimate and iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['cutoff_input'],
                         'Cutoff frequency (iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['event_direction_input'],
                         'Event direction (iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['replace_factor_input'],
                         'Factor for replacing events (multiple of event width, estimate and iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['analyse']['replace_gap_input'],
                         'Gap for replacing events (multiple of event width, estimate and iterate)', hover_delay=300)
        tooltip.Hovertip(self.widgets['load']['units'], 'Amplitude: pA, Duration: ms, Area: pC', hover_delay=300)

    def init_landing_page(self):
        # Landing page
        self.widgets['landing'] = dict()
        self.widgets['landing']['main'] = ttk.Frame(self.windows['main'], style='secondary.TFrame')
        self.widgets['landing']['main'].grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.widgets['landing']['inner'] = ttk.Frame(self.widgets['landing']['main'], style='primary.TFrame')
        self.widgets['landing']['main'].columnconfigure(0, weight=1)
        self.widgets['landing']['main'].rowconfigure(0, weight=1)
        self.widgets['landing']['inner'].grid(row=0, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

        # Information frame
        self.widgets['landing']['information'] = ttk.Frame(self.widgets['landing']['inner'])
        # Browse button
        self.widgets['landing']['button'] = ttk.Button(self.widgets['landing']['inner'], text="Browse", underline=0,
                                                       command=self.landing_load)

        # Layout
        self.widgets['landing']['inner'].columnconfigure(0, weight=1)
        self.widgets['landing']['inner'].rowconfigure(0, weight=1)
        self.widgets['landing']['information'].grid(row=0, column=0, sticky="nsew", padx=_WINDOW_PADDING,
                                                    pady=(_WINDOW_PADDING, 0))
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
        self.widgets['landing']['email_icon'].grid(row=1, column=0, sticky="nsew", padx=_PADDING, pady=(0, _PADDING))
        self.widgets['landing']['email'].grid(row=1, column=1, sticky="nsew", pady=(0, _PADDING))

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
        self.widgets['landing']['github_icon'].grid(row=1, column=1, sticky="nsew", padx=(0, _PADDING), pady=(0, _PADDING))
        self.widgets['landing']['github'].grid(row=1, column=2, sticky="nsew", padx=(0, _PADDING), pady=(0, _PADDING))

    def init_load(self):
        self.widgets['load'] = dict()
        # Load title
        self.widgets['load']['title'] = ttk.Label(self.windows['load'], text="Load",
                                                  style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['load_internal'] = ttk.Frame(self.windows['load'])
        # Layout
        self.windows['load'].columnconfigure(0, weight=1)
        self.widgets['load']['title'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0))
        self.windows['load'].rowconfigure(1, weight=1)
        self.windows['load_internal'].grid(row=1, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

        # Path information
        self.trace_information['path'] = tk.StringVar()
        self.trace_information['path'].set('')
        # Filename information
        self.widgets['load']['filename_label'] = ttk.Label(self.windows['load_internal'], text="Filename")
        self.trace_information['filename'] = tk.StringVar()
        self.trace_information['filename'].set('')
        wrap = (self.width // 6) - (7 * _PADDING) - self.fonts['small'].measure('Sample Rate (Hz)')
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
        self.widgets['load']['units_label'] = ttk.Label(self.windows['load_internal'], text="Input Units")
        self.trace_information['units'] = tk.StringVar()
        self.trace_information['units'].trace_add( 'write', lambda *args: self.update_units())
        self.trace_information['units'].set('pA')
        self.widgets['load']['units'] = ttk.Combobox(
            self.windows['load_internal'],
            values=['A', 'mA', 'µA', 'nA', 'pA', 'fA'],
            textvariable=self.trace_information['units'],
            state='readonly',
            width=_ENTRY_WIDTH,
            justify='center'
        )
        self.widgets['load']['units'].configure(font=self.fonts['small'])
        self.widgets['load']['units_separator'] = ttk.Separator(self.windows['load_internal'])
        # Range
        self.widgets['load']['range_label'] = ttk.Label(self.windows['load_internal'], text="Input Range (s, s)")
        self.widgets['load']['range_frame'] = ttk.Frame(self.windows['load_internal'])
        self.trace_information['range_start'] = tk.DoubleVar()
        self.trace_information['range_start'].set(0.0)
        self.widgets['load']['range_start'] = ttk.Spinbox(
            self.widgets['load']['range_frame'],
            textvariable=self.trace_information['range_start'],
            from_=0,
            to=0,
            increment=1,
            format='%4.6f',
            width=_ENTRY_WIDTH,
            justify='center',
            command=self.update_analysis_range
        )
        self.widgets['load']['range_start'].configure(font=self.fonts['small'])
        self.widgets['load']['range_start'].bind("<Return>", lambda _: self.update_analysis_range())
        self.widgets['load']['range_start'].bind("<FocusOut>", lambda _: self.update_analysis_range())
        self.trace_information['range_end'] = tk.DoubleVar()
        self.trace_information['range_end'].set(0.0)
        self.widgets['load']['range_end'] = ttk.Spinbox(
            self.widgets['load']['range_frame'],
            textvariable=self.trace_information['range_end'],
            from_=0,
            to=0,
            increment=1,
            format='%4.6f',
            width=_ENTRY_WIDTH,
            justify='center',
            command=self.update_analysis_range
        )
        self.widgets['load']['range_end'].configure(font=self.fonts['small'])
        self.widgets['load']['range_end'].bind("<Return>", lambda _: self.update_analysis_range())
        self.widgets['load']['range_end'].bind("<FocusOut>", lambda _: self.update_analysis_range())
        # Layout in frame
        self.widgets['load']['range_start'].grid(row=0, column=0, sticky="nsew", padx=(0, _PADDING))
        self.widgets['load']['range_end'].grid(row=0, column=1, sticky="nsew", padx=0)
        # Browse button
        self.widgets['load']['browse'] = ttk.Button(self.windows['load_internal'], text="Browse", underline=0,
                                                    command=self.load_trace, style='secondary.Outline.TButton')

        # Filename layout
        self.windows['load_internal'].columnconfigure(0, weight=1)
        self.windows['load_internal'].columnconfigure(1, weight=2)
        self.windows['load_internal'].rowconfigure(0, weight=1)
        self.widgets['load']['filename_label'].grid(row=0, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['load']['filename'].grid(row=0, column=1, sticky="e", padx=_PADDING)
        self.widgets['load']['filename_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        # Sample rate layout
        self.windows['load_internal'].rowconfigure(2, weight=1)
        self.widgets['load']['sample_rate_label'].grid(row=2, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['load']['sample_rate'].grid(row=2, column=1, sticky="e", padx=_PADDING)
        self.widgets['load']['sample_rate_separator'].grid(row=3, column=0, columnspan=2, sticky="nsew")
        # Samples layout
        self.windows['load_internal'].rowconfigure(4, weight=1)
        self.widgets['load']['samples_label'].grid(row=4, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['load']['samples'].grid(row=4, column=1, sticky="e", padx=_PADDING)
        self.widgets['load']['samples_separator'].grid(row=5, column=0, columnspan=2, sticky="nsew")
        # Duration layout
        self.windows['load_internal'].rowconfigure(6, weight=1)
        self.widgets['load']['duration_label'].grid(row=6, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['load']['duration'].grid(row=6, column=1, sticky="e", padx=_PADDING)
        self.widgets['load']['duration_separator'].grid(row=7, column=0, columnspan=2, sticky="nsew")
        # Units layout
        self.windows['load_internal'].rowconfigure(8, weight=1)
        self.widgets['load']['units_label'].grid(row=8, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['load']['units'].grid(row=8, column=1, sticky="e", padx=_PADDING)
        self.widgets['load']['units_separator'].grid(row=9, column=0, columnspan=2, sticky="nsew")
        # Range layout
        self.windows['load_internal'].rowconfigure(10, weight=1)
        self.widgets['load']['range_label'].grid(row=10, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['load']['range_frame'].grid(row=10, column=1, sticky="e", padx=_PADDING)
        # Browse layout
        self.widgets['load']['browse'].grid(row=11, column=0, columnspan=2, sticky="nsew", padx=_PADDING,
                                            pady=(0, _PADDING))

    def init_analyse(self):
        self.widgets['analyse'] = dict()
        # Options title
        self.widgets['analyse']['title'] = ttk.Label(self.windows['analyse'], text="Analyse",
                                                     style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['analyse_internal'] = ttk.Frame(self.windows['analyse'])
        # Layout
        self.windows['analyse'].columnconfigure(0, weight=1)
        self.widgets['analyse']['title'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0))
        self.windows['analyse'].rowconfigure(1, weight=1)
        self.windows['analyse_internal'].grid(row=1, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

        # Bounds filtering input
        self.analysis_options['_bounds_filter'] = tk.IntVar()
        self.analysis_options['_bounds_filter'].set(3)
        self.widgets['analyse']['bounds_label'] = ttk.Label(self.windows['analyse_internal'],
                                                            text="Bounds Filter (Samples)")
        self.widgets['analyse']['bounds_input'] = ttk.Spinbox(
            self.windows['analyse_internal'],
            from_=0,
            to=13,
            increment=1,
            textvariable=self.analysis_options['_bounds_filter'],
            format='%.0f',
            width=_ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['_bounds_filter'] = self.analysis_options['_bounds_filter'].get()
        self.widgets['analyse']['bounds_separator'] = ttk.Separator(self.windows['analyse_internal'])
        self.widgets['analyse']['bounds_input'].configure(font=self.fonts['small'])
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
            width=_ENTRY_WIDTH,
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
            width=_ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['z_score'] = self.analysis_options['z_score'].get()
        self.widgets['analyse']['z_score_input'].configure(font=self.fonts['small'])
        self.widgets['analyse']['z_score_separator'] = ttk.Separator(self.windows['analyse_internal'])
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
            width=_ENTRY_WIDTH,
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
            width=_ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['replace_gap'] = self.analysis_options['replace_gap'].get()
        self.widgets['analyse']['replace_gap_separator'] = ttk.Separator(self.windows['analyse_internal'])
        self.widgets['analyse']['replace_gap_input'].configure(font=self.fonts['small'])
        # Feature input
        self.analysis_options['features'] = tk.StringVar()
        self.analysis_options['features'].set('full')
        self.widgets['analyse']['features_label'] = ttk.Label(self.windows['analyse_internal'], text="Output Features")
        self.widgets['analyse']['features_input'] = ttk.Combobox(
            self.windows['analyse_internal'],
            values=['full', 'FWHM', 'FWQM'],
            textvariable=self.analysis_options['features'],
            state='readonly',
            width=_ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['features'] = self.analysis_options['features'].get()
        self.widgets['analyse']['features_separator'] = ttk.Separator(self.windows['analyse_internal'])
        self.widgets['analyse']['features_input'].configure(font=self.fonts['small'])
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
            width=_ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['estimate_cutoff'] = self.analysis_options['estimate_cutoff'].get()
        self.widgets['analyse']['estimate_cutoff_input'].configure(font=self.fonts['small'])
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
            width=_ENTRY_WIDTH,
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
            width=_ENTRY_WIDTH,
            justify='center'
        )
        self.default_analysis_options['event_direction'] = self.analysis_options['event_direction'].get()
        self.widgets['analyse']['event_direction_input'].configure(font=self.fonts['small'])
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
        self.widgets['analyse']['bounds_label'].grid(row=0, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['bounds_input'].grid(row=0, column=1, sticky="e", padx=_PADDING)
        self.widgets['analyse']['bounds_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        # Threshold window layout
        self.windows['analyse_internal'].rowconfigure(2, weight=1)
        self.widgets['analyse']['threshold_window_label'].grid(row=2, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['threshold_window_input'].grid(row=2, column=1, sticky="e", padx=_PADDING)
        self.widgets['analyse']['threshold_window_separator'].grid(row=3, column=0, columnspan=2, sticky="nsew")
        # Z-score layout
        self.windows['analyse_internal'].rowconfigure(4, weight=1)
        self.widgets['analyse']['z_score_label'].grid(row=4, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['z_score_input'].grid(row=4, column=1, sticky="e", padx=_PADDING)
        self.widgets['analyse']['z_score_separator'].grid(row=5, column=0, columnspan=2, sticky="nsew")
        # Replace factor layout
        self.windows['analyse_internal'].rowconfigure(6, weight=1)
        self.widgets['analyse']['replace_factor_label'].grid(row=6, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['replace_factor_input'].grid(row=6, column=1, sticky="e", padx=_PADDING)
        self.widgets['analyse']['replace_factor_separator'].grid(row=7, column=0, columnspan=2, sticky="nsew")
        # Replace gap layout
        self.windows['analyse_internal'].rowconfigure(8, weight=1)
        self.widgets['analyse']['replace_gap_label'].grid(row=8, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['replace_gap_input'].grid(row=8, column=1, sticky="e", padx=_PADDING)
        self.widgets['analyse']['replace_gap_separator'].grid(row=9, column=0, columnspan=2, sticky="nsew")
        # Features layout
        self.windows['analyse_internal'].rowconfigure(10, weight=1)
        self.widgets['analyse']['features_label'].grid(row=10, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['features_input'].grid(row=10, column=1, sticky="e", padx=_PADDING)
        self.widgets['analyse']['features_separator'].grid(row=11, column=0, columnspan=2, sticky="nsew")
        # Estimate cutoff layout
        self.windows['analyse_internal'].rowconfigure(12, weight=1)
        self.widgets['analyse']['estimate_cutoff_label'].grid(row=12, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['estimate_cutoff_input'].grid(row=12, column=1, sticky="e", padx=_PADDING)
        # Estimate button layout
        self.widgets['analyse']['estimate'].grid(row=13, column=0, columnspan=2, sticky="nsew", padx=_PADDING)
        # Cutoff layout
        self.windows['analyse_internal'].rowconfigure(14, weight=1)
        self.widgets['analyse']['cutoff_label'].grid(row=14, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['cutoff_input'].grid(row=14, column=1, sticky="e", padx=_PADDING)
        self.widgets['analyse']['cutoff_separator'].grid(row=15, column=0, columnspan=2, sticky="nsew")
        # Starting direction layout
        self.windows['analyse_internal'].rowconfigure(16, weight=1)
        self.widgets['analyse']['event_direction_label'].grid(row=16, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analyse']['event_direction_input'].grid(row=16, column=1, sticky="e", padx=_PADDING)
        # Iterate button layout
        self.widgets['analyse']['iterate'].grid(row=17, column=0, columnspan=2, sticky="nsew", pady=(0, _PADDING),
                                                padx=_PADDING)

    def init_analysis_view(self):
        self.widgets['analysis_view'] = dict()
        # Analysis view title
        self.widgets['analysis_view']['title'] = ttk.Label(self.windows['analysis_view'], text="Analysis View",
                                                           style='primary.Inverse.TLabel', anchor='center')
        # Internal frame for figure
        self.windows['analysis_view_figure'] = ttk.Frame(self.windows['analysis_view'])
        # Layout
        self.windows['analysis_view'].columnconfigure(0, weight=1)
        self.widgets['analysis_view']['title'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0))
        self.windows['analysis_view'].rowconfigure(1, weight=3)
        self.windows['analysis_view_figure'].grid(row=1, column=0, sticky="nsew", padx=_WINDOW_PADDING,
                                                  pady=(_WINDOW_PADDING, 0))

        # Create figure, axis and canvas
        self.widgets['analysis_view']['fig'] = Figure(layout='constrained')
        ax = self.widgets['analysis_view']['fig'].add_subplot(111)
        ax.set_xlabel("Time ($s$)", fontsize=_MEDIUM_FONT)
        ax.set_ylabel("Current" + f" (${self.trace_information['units'].get()}$)", fontsize=_MEDIUM_FONT)
        # Create canvas
        self.widgets['analysis_view']['canvas'] = CustomFigureCanvas(self.widgets['analysis_view']['fig'],
                                                                     self.windows['analysis_view_figure'], self,
                                                                     self.downsample_refresh)
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
        self.widgets['analysis_view']['toolbar'].grid(row=0, column=0, sticky="nsew", padx=_PADDING)
        self.widgets['analysis_view']['progress_bar'].grid(row=0, column=1, sticky='nse', padx=_PADDING, pady=_PADDING)
        self.widgets['analysis_view']['toolbar_separator'].grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.windows['analysis_view_figure'].rowconfigure(2, weight=1)
        self.widgets['analysis_view']['canvas'].get_tk_widget().grid(row=2, column=0, columnspan=2, sticky="nsew",
                                                                     padx=_PADDING, pady=_PADDING)

        # Bind events
        self.widgets['analysis_view']['canvas'].mpl_connect("key_press_event",
                                                            partial(self.toolbar_key_press, 'analysis_view'))
        self.widgets['analysis_view']['canvas'].mpl_connect("button_release_event", lambda _: self.update_figure_xlimits())
        self.widgets['analysis_view']['canvas'].mpl_connect("pick_event", self.pick_event)
        self.widgets['analysis_view']['canvas'].mpl_connect("button_press_event", self.press_analysis_range)
        self.widgets['analysis_view']['canvas'].mpl_connect("motion_notify_event", self.drag_analysis_range)
        self.widgets['analysis_view']['canvas'].mpl_connect("button_release_event", self.release_analysis_range)

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
        self.widgets['analysis_view']['options_toggle'].grid(row=2, column=0, sticky="ew", pady=(0, 0))
        self.windows['analysis_view'].rowconfigure(3, weight=1)
        self.windows['analysis_view_options'].grid(row=3, column=0, sticky="nsew", padx=_WINDOW_PADDING,
                                                   pady=(0, _WINDOW_PADDING))

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
        self.widgets['analysis_view']['reset'].grid(row=0, column=0, sticky="nsew", padx=(_PADDING, 0), pady=_PADDING)
        self.widgets['analysis_view']['lines'].grid(row=0, column=1, sticky="nsew", padx=_PADDING, pady=_PADDING)

        # Reset labelframe options
        items = [('all', self.reset_all), ('algorithm', self.reset_algorithm), ('figure', self.reset_figure),
                 ('results', self.reset_results)]
        self.widgets['analysis_view']['reset'].columnconfigure(0, weight=1, uniform='options')
        for i, item in enumerate(items):
            self.widgets['analysis_view'][f'reset_{item[0]}_button'] = ttk.Button(
                self.widgets['analysis_view']['reset'],
                text=item[0].capitalize(),
                width=2 * _ENTRY_WIDTH,
                command=item[1],
                style='Small.secondary.Outline.TButton'
            )
            self.widgets['analysis_view']['reset'].rowconfigure(i, weight=1, uniform='options')
            self.widgets['analysis_view'][f'reset_{item[0]}_button'].grid(row=i, column=0, padx=_PADDING)

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
            self.widgets['analysis_view'][f'{option[0]}_label'].grid(row=0, column=i + 1, sticky="nsew", padx=_PADDING)
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
            self.widgets['analysis_view'][f'{key[0]}_label'].grid(row=i, column=0, sticky="nsew", padx=_PADDING)
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
                        width=_ENTRY_WIDTH
                    )
                    self.widgets['analysis_view'][option[0]][key[0]].bind("<Return>", lambda _, f = option[2], k = key[0]: f(k))
                    self.widgets['analysis_view'][option[0]][key[0]].bind("<FocusOut>", lambda _, f = option[2], k = key[0]: f(k))
                    self.widgets['analysis_view'][option[0]][key[0]].configure(font=self.fonts['small'])
                elif option[1] == 'colour':
                    self.figure_options[option[0]][key[0]] = tk.StringVar()
                    self.widgets['analysis_view'][option[0]][key[0]] = ttk.Button(
                        self.widgets['analysis_view']['lines'],
                        command=partial(option[2],key[0]),
                        width=_ENTRY_WIDTH + 2
                    )
                elif option[1] == 'str':
                    self.figure_options[option[0]][key[0]] = tk.StringVar()
                    self.widgets['analysis_view'][option[0]][key[0]] = ttk.Combobox(
                        self.widgets['analysis_view']['lines'],
                        textvariable=self.figure_options[option[0]][key[0]],
                        state='readonly',
                        justify='center',
                        width=_ENTRY_WIDTH
                    )
                    self.widgets['analysis_view']['style'][key[0]].bind('<<ComboboxSelected>>', lambda _, f = option[2], k = key[0]: f(k))
                    self.widgets['analysis_view'][option[0]][key[0]].configure(font=self.fonts['small'])

                # Specific configuration
                if option[0] == 'show':
                    self.figure_options[option[0]][key[0]].set(True if key[0] in ['trace', 'baseline', 'threshold'] else
                                                               False)
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=(_PADDING + 3, _PADDING),
                                                                          pady=(1, 0))
                elif option[0] == 'linewidth':
                    self.figure_options[option[0]][key[0]].set(1.0)
                    self.widgets['analysis_view'][option[0]][key[0]].configure(from_=0, to=4, increment=0.1)
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=_PADDING)
                elif option[0] == 'colour':
                    self.figure_options[option[0]][key[0]].set(self.colours[key[0]])
                    self.style.configure(f'{key[0]}_colour.TButton', background=self.colours[key[0]],
                                         borderwidth=0, font=self.fonts['small_b'])
                    self.style.map(f'{key[0]}_colour.TButton', background=[('active', self.colours[key[0]])])
                    self.widgets['analysis_view'][option[0]][key[0]].configure(style=f'{key[0]}_colour.TButton')
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=_PADDING)
                elif option[0] == 'style':
                    self.figure_options[option[0]][key[0]].set('-')
                    self.widgets['analysis_view'][option[0]][key[0]].configure(values=['-', '--', ':', '-.'])
                    self.widgets['analysis_view'][option[0]][key[0]].grid(row=i, column=j, padx=_PADDING)


                self.default_figure_options[option[0]][key[0]] = self.figure_options[option[0]][key[0]].get()
                j += 1

            i += 2

        # Extra options
        self.widgets['analysis_view']['extra_options'] = ttk.Frame(self.widgets['analysis_view']['lines'])
        self.widgets['analysis_view']['extra_options'].grid(row=0, column=0, sticky="nw", padx=_PADDING)

        # Add grid to the top left corner
        self.figure_options['grid'] = tk.BooleanVar(value=False)
        self.default_figure_options['grid'] = self.figure_options['grid'].get()
        self.widgets['analysis_view']['grid'] = ttk.Button(
            self.widgets['analysis_view']['extra_options'],
            text='Grid',
            width=_ENTRY_WIDTH,
            command=self.update_figure_grid,
            style='Small.secondary.Outline.TButton',
        )
        self.widgets['analysis_view']['grid'].grid(row=0, column=0, sticky="nw", padx=(0, _PADDING))

        # Add baseline subtraction toggle
        self.figure_options['baseline_subtract'] = tk.BooleanVar(value=False)
        self.default_figure_options['baseline_subtract'] = self.figure_options['baseline_subtract'].get()
        self.widgets['analysis_view']['baseline_subtract'] = ttk.Button(
            self.widgets['analysis_view']['extra_options'],
            text='Detrend',
            width=_ENTRY_WIDTH,
            command=self.update_figure_baseline_subtract,
            style='Small.secondary.Outline.TButton',
        )
        self.widgets['analysis_view']['baseline_subtract'].grid(row=0, column=1, sticky="ne", padx=0)

    def init_results(self):
        self.widgets['results'] = dict()
        # Results title
        self.widgets['results']['title'] = ttk.Label(self.windows['results'], text="Results",
                                                     style='primary.Inverse.TLabel', anchor='center')
        # Internal frame
        self.windows['results_internal'] = ttk.Frame(self.windows['results'])
        # Layout
        self.windows['results'].columnconfigure(0, weight=1)
        self.widgets['results']['title'].grid(row=0, column=0, sticky="nsew", pady=(_WINDOW_PADDING, 0))
        self.windows['results'].rowconfigure(1, weight=1)
        self.windows['results_internal'].grid(row=1, column=0, sticky="nsew", padx=_WINDOW_PADDING, pady=_WINDOW_PADDING)

        # Results notebook
        self.widgets['results']['notebook'] = ttk.Notebook(self.windows['results_internal'])
        self.widgets['results']['notebook'].enable_traversal()
        self.widgets['results']['notebook'].bind("<<NotebookTabChanged>>", self.notebook_tab_changed)
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
        self.widgets['results']['save'].grid(row=1, column=0, sticky="nsew", padx=_PADDING, pady=_PADDING)


    """
    Reset
    """

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
            if isinstance(self.trace_information[key], tk.StringVar):
                self.trace_information[key].set('')
            elif isinstance(self.trace_information[key], tk.DoubleVar):
                self.trace_information[key].set(0.0)
        # Set the flags
        self.flags['loaded'] = False
        # Update options
        self.update_options()

    def reset_algorithm(self):
        for key in self.default_analysis_options:
            self.analysis_options[key].set(self.default_analysis_options[key])

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
        # Clear the results
        self.analysis_results = dict()
        # Destroy feature window
        if self.widgets['results']['feature_window']:
            self.widgets['results']['feature_window'].destroy()
        # Update the results
        self.update_results()
        # Update the figure
        self.update_figure(title=' ', retain_view=False)
        self.figure_options['baseline_subtract'].set(False)
        # Set the flags
        self.flags['estimated'] = False
        self.flags['iterated'] = False
        self.flags['saved'] = False
        # Update options
        self.update_options()

    def reset_all(self):
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        # Reset the load window
        self.reset_load()
        # Reset the analyse window
        self.reset_algorithm()
        # Reset the results window
        self.reset_results()
        # Reset the analysis view window
        self.reset_figure()


    """
    Update
    """

    def update_load(self, path):
        # Path
        self.trace_information['path'].set(path)
        if path != '':
            if len(basename(path)) > 63:
                self.trace_information['filename'].set(f'{basename(path)[0:60]}...')
            else:
                self.trace_information['filename'].set(basename(path))

        if type(self.trace) is not np.ndarray or type(self.time_vector) is not np.ndarray:
            return

        # General trace information
        self.trace_information['samples'].set(f'{len(self.trace):.0f}')
        self.trace_information['sample_rate'].set(f'{1 / self.time_step:.0f}')
        self.trace_information['duration'].set(f'{self.time_vector[len(self.time_vector) - 1] - self.time_vector[0]:.1f}')

        # Analysis range
        start_time = float(self.time_vector[0])
        stop_time = float(self.time_vector[len(self.time_vector) - 1])
        step = len(self.time_vector) * float(self.time_step) / 20
        self.trace_information['range_start'].set(start_time)
        self.widgets['load']["range_start"].configure(from_=start_time, to=stop_time, increment=step)
        self.trace_information['range_end'].set(stop_time)
        self.widgets['load']["range_end"].configure(from_=start_time, to=stop_time, increment=step)
        self.current_range = (0, len(self.trace))
        self.pending_range = None

    def update_z_score(self):
        if type(self.trace) is not np.ndarray or type(self.time_vector) is not np.ndarray:
            return

        # Set the z-score
        start_idx, end_idx = self.current_range or (0, len(self.trace))
        trace = self.trace[start_idx:end_idx]

        self.analysis_options['z_score'].set(np.round(norm.ppf(1.0 - ((1.0 / len(trace)) / 2.0)), 3))
        self.default_analysis_options['z_score'] = self.analysis_options['z_score'].get()
        self.flash_entry(self.widgets['analyse']['z_score_input'])

    def update_units(self):
        units = self.trace_information['units'].get()
        if 'analysis_view' not in self.widgets:
            return
        if not units:
            return

        # Update y label
        ax = self.widgets['analysis_view']['fig'].axes[0]
        if self.figure_options['baseline_subtract'].get():
            current = self.analysis_results.get(self.last_called)
            if type(getattr(current, "baseline", None)) is np.ndarray:
                ax.set_ylabel("Current - Baseline" + f" (${units}$)", fontsize=_MEDIUM_FONT)
        else:
            ax.set_ylabel("Current" + f" (${units}$)", fontsize=_MEDIUM_FONT)

        # Update tooltip
        if units == 'A':
            tooltip.Hovertip(self.widgets['load']['units'], f'Amplitude: A, Duration: ms, Area: C',
                             hover_delay=300)
        else:
            tooltip.Hovertip(self.widgets['load']['units'], f'Amplitude: {units[0]}A, Duration: ms, Area: {units[0]}C',
                             hover_delay=300)

        # Update feature window if it exists
        if self.widgets['results']['feature_window'] and self.widgets['results']['feature_window'].winfo_exists():
                self.widgets['results']['feature_window'].update_all_figures(relim=True)

        # Redraw
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def update_analysis_range(self):
        if type(self.trace) is not np.ndarray or type(self.time_vector) is not np.ndarray:
            return

        # Full range and user range
        full_start = float(self.time_vector[0])
        full_end = float(self.time_vector[len(self.time_vector) - 1])
        start_time = self.trace_information['range_start'].get()
        end_time = self.trace_information['range_end'].get()

        # Clamp to full range and assert ordering
        start_time = max(full_start, min(start_time, full_end))
        end_time = max(full_start, min(end_time, full_end))
        if start_time > end_time:
            start_time, end_time = end_time, start_time

        # Get indices of user range
        start_idx = int(np.searchsorted(self.time_vector, start_time, side='left'))
        end_idx = int(np.searchsorted(self.time_vector, end_time, side='right'))

        # Clamp to full indices and ensure ten steps (Shapiro)
        start_idx = max(0, min(start_idx, len(self.trace) - 10))
        end_idx = max(start_idx + 10, min(end_idx, len(self.trace)))

        # Set current range and update entries
        self.current_range = (start_idx, end_idx)
        self.trace_information['range_start'].set(float(self.time_vector[start_idx]))
        self.trace_information['range_end'].set(float(self.time_vector[end_idx - 1]))

        # Plot the lines
        self.update_analysis_range_lines()

        # Update z-score
        self.update_z_score()

    def update_analysis_range_lines(self):
        # Check if the figure is initialised
        if self.widgets['analysis_view']['fig'].axes:
            ax = self.widgets['analysis_view']['fig'].axes[0]
        else:
            return

        # Check if time vector and time step are initialised
        if type(self.time_vector) is not np.ndarray or not self.time_step:
            return

        # Remove old lines
        for line in list(ax.lines):
            if line.get_label() in ['_range_start', '_range_end']:
                line.remove()

        # Plot lines
        ax.axvline(self.trace_information['range_start'].get(), color=self.colours['baseline'], linestyle='--',
                   linewidth=2, zorder=100, label='_range_start')
        ax.axvline(self.trace_information['range_end'].get(), color=self.colours['baseline'], linestyle='--',
                   linewidth=2, zorder=100, label='_range_end')

        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

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
            xlim = ax.get_xlim() if retain_view else (self.time_vector[0], self.time_vector[len(self.time_vector) - 1])

            # Plot the trace
            if type(self.trace) is np.ndarray and self.figure_options['show']['trace'].get():
                ax.plot(*self.downsample_line(self.time_vector, self.trace, xlim), label='trace')

            # Current results
            current = self.analysis_results.get(self.last_called)

            # Plot the calculation trace
            if type(getattr(current, "calculation_trace", None)) is np.ndarray and self.figure_options['show']['calculation_trace'].get():
                ax.plot(*self.downsample_line(self.time_vector, current.calculation_trace, xlim), label='calculation_trace')

            # Plot the filtered trace
            if type(getattr(current, "filtered_trace", None)) is np.ndarray and self.figure_options['show']['filtered_trace'].get():
                ax.plot(*self.downsample_line(self.time_vector, current.filtered_trace, xlim), label='filtered_trace')

            # Plot the baseline
            if type(getattr(current, "baseline", None)) is np.ndarray and self.figure_options['show']['baseline'].get():
                ax.plot(*self.downsample_line(self.time_vector, current.baseline, xlim), label='baseline')

            # Plot the threshold
            if (type(getattr(current, "pos_threshold", None)) is np.ndarray and type(getattr(current, "baseline", None))
                    is np.ndarray and self.figure_options['show']['threshold'].get()):
                ax.plot(*self.downsample_line(self.time_vector, current.pos_threshold, xlim), label='threshold')
                ax.plot(*self.downsample_line(self.time_vector, current.neg_threshold, xlim), label='threshold')

            # Plot the events
            if type(getattr(current, "event_coordinates", None)) is np.ndarray and self.figure_options['show']['events'].get():
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

        # Set title
        if title:
            ax.set_title(title, fontsize=_LARGE_FONT)

        # Set labels and grid
        ax.set_xlabel("Time ($s$)", fontsize=_MEDIUM_FONT)
        units = self.trace_information['units'].get() or 'pA'
        if self.figure_options['baseline_subtract'].get() and type(getattr(current, "baseline", None)) is np.ndarray:
            ax.set_ylabel("Current - Baseline" + f" (${units}$)", fontsize=_MEDIUM_FONT)
        else:
            ax.set_ylabel("Current" + f" (${units}$)", fontsize=_MEDIUM_FONT)
        ax.grid(self.figure_options['grid'].get())
        self.update_analysis_range_lines()
        self.widgets['analysis_view']['canvas'].draw_idle()
        self.widgets['analysis_view']['canvas'].flush_events()

    def update_figure_grid(self):
        grid = self.figure_options['grid'].get()
        self.figure_options['grid'].set(not grid)
        self.widgets['analysis_view']['fig'].axes[0].grid(not grid)
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def update_figure_baseline_subtract(self):
        current = self.analysis_results.get(self.last_called)
        if current and type(current.baseline) is not np.ndarray:
            return

        subtract = self.figure_options['baseline_subtract'].get()
        self.figure_options['baseline_subtract'].set(not subtract)
        self.update_figure(retain_view=False)

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
            current = self.analysis_results.get(self.last_called)
            if key == 'trace' and type(self.trace) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, self.trace, xlim), label='trace'))
            elif key == 'calculation_trace' and type(current.calculation_trace) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, current.calculation_trace, xlim), label='calculation_trace'))
            elif key == 'filtered_trace' and type(current.filtered_trace) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, current.filtered_trace, xlim), label='filtered_trace'))
            elif key == 'baseline' and type(current.baseline) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, current.baseline, xlim), label='baseline'))
            elif key == 'threshold' and type(current.pos_threshold) is np.ndarray and type(current.baseline) is np.ndarray:
                lines.append(ax.plot(*self.downsample_line(self.time_vector, current.pos_threshold, xlim), label='threshold'))
                lines.append(ax.plot(*self.downsample_line(self.time_vector, current.neg_threshold, xlim), label='threshold'))
            elif key == 'events' and type(current.event_coordinates) is np.ndarray:
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

    def update_figure_xlimits(self, limits = None, push = False):
        if limits:
            self.after_id = None
            ax = self.widgets['analysis_view']['fig'].axes[0]
            ax.set_xlim(limits)
            # Downsample all lines
            self.downsample_all()
            # Push state to toolbar
            if push is True and np.all(np.round(limits, 5) == np.round(ax.get_xlim(), 5)):
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
        if 'Iterate' in self.analysis_results.keys():
            tabs.append(self.create_tab(self.analysis_results['Iterate'], 'Iterate'))

        # Add the statistics tab
        if 'Statistics' in self.analysis_results.keys():
            tab = self.create_statistics_tab()
            if tab:
                tabs.append(tab)

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
        if self.widgets['results']['feature_window'] and self.widgets['results']['feature_window'].winfo_exists():
            self.widgets['results']['feature_window'].update_events(self.analysis_results[self.last_called].events)

        # Set the last called tab
        if self.widgets['results']['notebook'].tabs():
            for frame, title in tabs:
                if title == self.last_called:
                    self.widgets['results']['notebook'].select(frame)
                    break
            else:
                self.widgets['results']['notebook'].select(0)
            self.toggle_results(True)
        else:
            self.toggle_results(False)

    def update_options(self):
        if self.flags['estimated'] or self.flags['iterated']:
            state = 'estimated'
        elif self.flags['loaded']:
            state = 'loaded'
        else:
            state = 'reset'

        loaded = state in ('loaded', 'estimated')
        estimated = state == 'estimated'

        # Input options available once a trace is loaded
        unloaded_options = ['units', 'range_start', 'range_end']
        for key in unloaded_options:
            widget = self.widgets['load'][key]
            state = widget.state()
            if 'readonly' in state or '!readonly' in state:
                widget.state(['!disabled', 'readonly'] if loaded else ['disabled'])
            else:
                widget.state(['!disabled'] if loaded else ['disabled'])

        # Main analysis options available once a trace is loaded
        loaded_options = ['bounds_label', 'bounds_input', 'threshold_window_label', 'threshold_window_input', 'z_score_label',
                          'z_score_input', 'replace_factor_label', 'replace_factor_input', 'replace_gap_label', 'replace_gap_input',
                          'features_label', 'features_input', 'estimate_cutoff_label', 'estimate_cutoff_input', 'estimate',]
        for key in loaded_options:
            widget = self.widgets['analyse'][key]
            state = widget.state()
            if 'readonly' in state or '!readonly' in state:
                widget.state(['!disabled', 'readonly'] if loaded else ['disabled'])
            else:
                widget.state(['!disabled'] if loaded else ['disabled'])

        # Iterate options available only after estimate
        iterate_options = ['event_direction_label', 'event_direction_input', 'cutoff_input', 'cutoff_label', 'iterate']
        for key in iterate_options:
            widget = self.widgets['analyse'][key]
            state = widget.state()
            if 'readonly' in state or '!readonly' in state:
                widget.state(['!disabled', 'readonly'] if estimated else ['disabled'])
            else:
                widget.state(['!disabled'] if estimated else ['disabled'])

        # Results save button available only after estimate/iteration
        self.widgets['results']['save'].state(['!disabled'] if estimated else ['disabled'])

        # Reset buttons
        for key in ['all', 'algorithm', 'figure', 'results']:
            enabled = estimated if key == 'results' else loaded
            self.widgets['analysis_view'][f'reset_{key}_button'].state(['!disabled'] if enabled else ['disabled'])

        # Figure option buttons
        self.widgets['analysis_view']['grid'].state(['!disabled'] if loaded else ['disabled'])
        self.widgets['analysis_view']['baseline_subtract'].state(['!disabled'] if estimated else ['disabled'])

        # Line controls
        result_lines = {'baseline', 'threshold', 'calculation_trace', 'filtered_trace', 'events', 'trace'}
        for option in ['show', 'linewidth', 'colour', 'style']:
            for key in result_lines:
                enabled = loaded if key == 'trace' else estimated
                widget = self.widgets['analysis_view'][option][key]
                state = widget.state()
                if 'readonly' in state or '!readonly' in state:
                    widget.state(['!disabled', 'readonly'] if enabled else ['disabled'])
                else:
                    widget.state(['!disabled'] if enabled else ['disabled'])

    """
    Toggle    
    """

    def hide(self):
        self.attributes('-alpha', 0)

    def show(self):
        self.attributes('-alpha', 0)
        self.deiconify()
        self.update()
        self.widgets['analysis_view']['canvas'].resize(instant=True)
        self.update()
        self.attributes('-alpha', 1)

    def toggle_landing_page(self, show: Optional[bool] = None):
        if show is None:
            show = False if self.widgets['landing']['main'].winfo_manager() else True

        self.hide()
        if self.widgets['landing']['main'].winfo_manager() and show is False:
            # Show windows
            self.widgets['landing']['main'].grid_forget()
            self.geometry(self.default_geometry)
            self.state('zoomed')
            self.resizable(True, True)
            self.toggle_load(True)
            self.toggle_analyse(True)
            self.toggle_analysis_view(True)
            self.widgets['analyse']['estimate'].focus_set()
        elif show is True:
            # Hide all windows
            self.toggle_load(False)
            self.toggle_analyse(False)
            self.toggle_analysis_view(False)
            self.toggle_analysis_view_options(False)
            self.toggle_results(False)
            self.widgets['landing']['main'].grid(row=0, column=0, sticky="nsew")
            self.geometry(self.landing_geometry)
            self.state('normal')
            self.resizable(False, False)
        self.show()

    def toggle_pane(self, pane_key: str, window_key: str, show: Optional[bool], row: int, column: int, sticky: str,
                    padx: tuple[int, int] | int, pady: tuple[int, int] | int, weight: int, keep: tuple[str, ...]):
        pane = self.windows[pane_key]
        window = self.windows[window_key]

        if show is None:
            show = False if window.winfo_manager() else True

        # Hide the window and pane if necessary
        if show is False:
            window.grid_forget()
            if not any(self.windows[key].winfo_manager() for key in keep):
                if str(pane) in self.windows['paned'].panes():
                    self.windows['paned'].forget(pane)
            return
        
        # Show the window and pane
        if str(pane) not in self.windows['paned'].panes():
            panes = list(self.windows['paned'].panes())
            if pane_key == 'left':
                if panes:
                    self.windows['paned'].insert(0, pane, weight=weight)
                else:
                    self.windows['paned'].add(pane, weight=weight)
            elif pane_key == 'center':
                if str(self.windows['right']) in panes:
                    self.windows['paned'].insert(panes.index(str(self.windows['right'])), pane, weight=weight)
                else:
                    self.windows['paned'].add(pane, weight=weight)
            elif pane_key == 'right':
                self.windows['paned'].add(pane, weight=weight)
        window.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)

    def toggle_load(self, show: Optional[bool] = None):
        self.toggle_pane(
            pane_key='left',
            window_key='load',
            show=show,
            row=0,
            column=0,
            sticky='nsew',
            pady=(_WINDOW_PADDING, 0),
            padx=(_WINDOW_PADDING, 0),
            weight=0,
            keep=('analyse',)
        )

    def toggle_analyse(self, show: Optional[bool] = None):
        self.toggle_pane(
            pane_key='left',
            window_key='analyse',
            show=show,
            row=1,
            column=0,
            sticky='nsew',
            pady=_WINDOW_PADDING,
            padx=(_WINDOW_PADDING, 0),
            weight=0,
            keep=('load',)
        )

    def toggle_analysis_view(self, show: Optional[bool] = None):
        self.toggle_pane(
            pane_key='center',
            window_key='analysis_view',
            show=show,
            row=0,
            column=0,
            sticky='nsew',
            pady=_WINDOW_PADDING,
            padx=0,
            weight=1,
            keep=()
        )

        # Instant call to resize
        self.widgets['analysis_view']['canvas'].resize(instant=True)

    def toggle_results(self, show: Optional[bool] = None):
        self.toggle_pane(
            pane_key='right',
            window_key='results',
            show=show,
            row=0,
            column=0,
            sticky='nsew',
            pady=_WINDOW_PADDING,
            padx=(0, _WINDOW_PADDING),
            weight=0,
            keep=()
        )

    def toggle_analysis_view_options(self, show: Optional[bool] = None):
        if show is None:
            show = False if self.windows['analysis_view_options'].winfo_manager() else True

        if self.windows['analysis_view_options'].winfo_manager() and show is False:
            self.windows['analysis_view'].rowconfigure(3, weight=0)
            self.windows['analysis_view_options'].grid_forget()
        elif show is True:
            self.windows['analysis_view'].rowconfigure(3, weight=1)
            self.windows['analysis_view_options'].grid(row=3, column=0, sticky="nsew", padx=_WINDOW_PADDING,
                                                       pady=(0, _WINDOW_PADDING))


    """
    Runtime init
    """

    def create_feature_window(self):
        if self.flags['feature_win']:
            return
        self.flags['feature_win'] = True

        tab_id = self.widgets['results']['notebook'].select()
        text =  self.widgets['results']['notebook'].tab(tab_id, "text")
        if text == 'Statistics':
            text = 'Iterate'

        if len(self.analysis_results[text].events) == 0:
            self.flags['feature_win'] = False
            return

        if self.widgets['results']['feature_window'] and self.widgets['results']['feature_window'].winfo_exists():
            self.widgets['results']['feature_window'].update_events(self.analysis_results[text].events)
        else:
            self.widgets['results']['feature_window'] = FeatureWindow(self, self.analysis_results[text].events)
        self.widgets['results']['feature_window'].show()

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
        feature_window_button.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=_PADDING, pady=_PADDING)
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
            labels[len(labels) - 1].grid(row=row, column=0, sticky="nsew", padx=_PADDING)
            values.append(ttk.Label(stats_frame, text=f'{main_stats[stat]:.6g}'))
            values[len(values) - 1].grid(row=row, column=1, sticky="e", padx=_PADDING)
            if i != len(main_stats) - 1:
                seperators.append(ttk.Separator(stats_frame))
                seperators[len(seperators) - 1].grid(row=row + 1, column=0, columnspan=2, sticky="nsew")
            row += 2

        # Populate the event statistics
        event_stats = results.events.events
        if not event_stats:
            return tab, title

        first_inner = next(iter(event_stats.values()))
        keys = list(first_inner.keys())
        table_stats = [{} for i in range(len(event_stats))]
        max_width = dict()
        # Format
        for key in keys:
            if key not in self.table_columns:
                continue

            for i, stat in enumerate(event_stats.values()):
                if key == 'Coordinates':
                    table_stats[i][key] = f'({stat[key][0]}, {stat[key][1]})'
                else:
                    table_stats[i][key] = f'{np.round(stat[key], 2):g}'

            row_width = max([len(stat[key]) for stat in table_stats]) + 2
            row_width *= self.fonts['small'].measure("0")
            heading_width = len(str(key)) + 2
            heading_width *= self.fonts['small_i'].measure("0")
            max_width[key] = row_width if row_width > heading_width else heading_width
        for i, stat in enumerate(event_stats.values()):
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

    def create_statistics_tab(self):
        if 'Statistics' in self.analysis_results.keys():
            trace_stats = self.analysis_results['Statistics']['trace_stats']
            event_stats = self.analysis_results['Statistics']['event_stats']
        else:
            return
        if len(trace_stats) < 2 or len(event_stats) < 2:
            return

        # Create tab and frames
        tab = ttk.Frame(self.widgets['results']['notebook'])
        header_separator = ttk.Separator(tab, style='primary.TSeparator')
        graph_frame = ttk.Frame(tab)

        # Tab layout
        tab.columnconfigure(0, weight=1)
        header_separator.grid(row=0, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        graph_frame.grid(row=1, column=0, sticky="nsew", pady=(0, _PADDING))
        footer_separator = ttk.Separator(tab, style='primary.TSeparator')
        footer_separator.grid(row=2, column=0, sticky="nsew")

        # Graph frame
        length = len(trace_stats[0].keys()) + len(event_stats[0].keys())
        fig = Figure(layout='constrained', figsize=(1, 1.5 * length))
        # Trace stats
        x = np.arange(1, len(trace_stats) + 1)
        for i, key in enumerate(trace_stats[0].keys()):
            ax = fig.add_subplot(length, 1, i + 1)
            ax.axvline(x=len(trace_stats), color=self.colours['baseline'], linestyle='--', linewidth=0.5)
            ax.plot(x, [stats[key] for stats in trace_stats], label=key, color=self.colours['trace'],
                    linewidth=0.5, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
            ax.scatter(x, [stats[key] for stats in trace_stats], label=key, color=self.colours['threshold'],
                       path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
            ax.set_title(key, fontsize=_SMALL_FONT)
            ax.tick_params(axis='both', which='both', labelsize=_SMALL_FONT - 2)
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
            ax.set_title(key, fontsize=_SMALL_FONT)
            ax.tick_params(axis='both', which='both', labelsize=_SMALL_FONT - 2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.supxlabel('Iteration', fontsize=_SMALL_FONT)
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
        inner_window = inner_canvas.create_window(0, 0, window=inner_frame, anchor=tk.NW)

        # Do this on configure so that the canvas resizes
        def resize_inner_frame(event):
            current_width = inner_canvas.itemcget(inner_window, "width")
            if current_width and int(float(current_width)) == event.width:
                return

            inner_canvas.itemconfigure(inner_window, width=event.width)
            inner_canvas.configure(scrollregion=inner_canvas.bbox("all"))
        inner_canvas.bind("<Configure>", resize_inner_frame)

        # Plot inside inner frame... inside inner canvas
        canvas = CustomFigureCanvas(fig, inner_frame, self)
        self.widgets['results']['statistics_canvas'] = canvas
        canvas.draw()

        # Bind this so scrollbar propagates
        canvas.get_tk_widget().bind("<MouseWheel>", lambda e: inner_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"),
                                    add="+")

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
        toolbar.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=_PADDING)
        toolbar_separator = ttk.Separator(graph_frame)
        toolbar_separator.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, _PADDING))
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw_idle()
        canvas.flush_events()

        return tab, 'Statistics'


    """
    Events
    """

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

        # Ignore mouse events on headings, separators
        if getattr(event, "type", None) in (tk.EventType.ButtonPress, tk.EventType.ButtonRelease):
            region = tree.identify_region(event.x, event.y)
            if region not in ("cell", "tree"):
                return

            item = tree.identify_row(event.y)
            if not item:
                return

            current_selection = tree.selection()
            if item not in current_selection:
                tree.selection_set(item)
                tree.focus(item)
            else:
                tree.focus(item)

        selection = tree.selection()
        if not selection:
            return

        current = self.analysis_results[title]
        events = current.events
        for item in selection:
            event_id = int(tree.item(item, 'text'))

            # Toggle the outlier flag
            if events.get(event_id)['Outlier']:
                events.get(event_id)['Outlier'] = False
                tree.item(item, tags='not_outlier')
            else:
                events.get(event_id)['Outlier'] = True
                tree.item(item, tags='outlier')

        event_coordinates = []
        for event in events:
            if not event['Outlier']:
                event_coordinates.append(event['Coordinates'])

        # Recalculate cutoff
        try:
            current.event_stats['Max Cutoff'] = np.round(
                nn.methods._calculate_cutoffs(
                    self.trace,
                    current.baseline,
                    current.initial_threshold,
                    np.asarray(event_coordinates, dtype=np.int64),
                    current.cutoff,
                    int(1 / self.time_step)
                )[0],
                2
            )
        except IndexError:
            current.event_stats['Max Cutoff'] = self.analysis_options['estimate_cutoff'].get()

        if title == self.last_called:
            self.analysis_options['cutoff'].set(current.event_stats['Max Cutoff'])
            self.flash_entry(self.widgets['analyse']['cutoff_input'])

        # Update the feature window figure
        if self.widgets['results']['feature_window']:
            self.widgets['results']['feature_window'].update_all_figures()

    def jump_to_event(self, title, event):
        # Get the event ID
        tree = event.widget

        # Ignore mouse events on headings, separators
        if getattr(event, "type", None) in (tk.EventType.ButtonPress, tk.EventType.ButtonRelease):
            region = tree.identify_region(event.x, event.y)
            if region not in ("cell", "tree"):
                return

            item = tree.identify_row(event.y)
            if not item:
                return

            tree.selection_set(item)
            tree.focus(item)

        else:
            # Keyboard
            item = tree.focus()

            if not item:
                selection = tree.selection()
                if not selection:
                    return
                item = selection[0]

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

        # Whether baseline subtracted
        if self.figure_options['baseline_subtract'].get() and type(self.analysis_results[self.last_called].baseline) is np.ndarray:
            trace = self.trace - self.analysis_results[self.last_called].baseline
        else:
            trace = self.trace

        # Calculate new limits
        ax = self.widgets['analysis_view']['fig'].axes[0]
        visible_trace = trace[start:end + 1]
        finite = np.isfinite(visible_trace)
        if not finite.any():
            return
        trace_min = np.min(visible_trace[finite])
        trace_max = np.max(visible_trace[finite])
        difference = 0.5 * abs(trace_max - trace_min)
        if difference == 0:
            difference = 1

        # Update limits
        old_xlim = ax.get_xlim()
        new_xlim = (self.time_vector[start], self.time_vector[end])
        self.update_figure_xlimits((self.time_vector[start], self.time_vector[end]), push=False)
        ax.set(ylim=(trace_min - difference, trace_max + difference))

        # Plot flash
        line = ax.plot(self.time_vector[event_coordinates[0]:event_coordinates[1] + 1],
                       trace[event_coordinates[0]:event_coordinates[1] + 1], 'r', linewidth=3, zorder=8)

        # Only push to toolbar if different
        if not np.allclose(old_xlim, new_xlim):
            self.widgets['analysis_view']['toolbar'].push_current()
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()
        self.after(500, remove)

    def pick_event(self, event):
        # Convert point to event coordinates
        point = np.round((event.artist.get_xdata()[event.ind[0]] - self.time_vector[0]) / self.time_step)

        # Get currently shown events
        events = self.analysis_results[self.last_called].events

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
            if self.widgets['results']['notebook'].tab(i, "text") == self.last_called:
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
                self.jump_to_event(self.last_called, dummy)
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
        if self.flashing.get(entry) is not None:
            return
        self.flashing[entry] = entry

        current = entry.cget('style')
        flash = f'flash.{current}'

        self.style.configure(flash, bordercolor=self.style.colors.primary, foreground=self.style.colors.primary)
        for i in np.arange(0, 10, 2):
            self.after(i * 250, lambda e=entry, s=flash: e.configure(style=s))
            self.after((i + 1) * 250, lambda e=entry, s=current: e.configure(style=s))
        self.after(3000, lambda e=entry, s=current: (e.configure(style=s), self.flashing.pop(e)))

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

    def notebook_tab_changed(self, event):
        canvas = self.widgets['results'].get('statistics_canvas')
        notebook = self.widgets['results'].get('notebook')
        if canvas and notebook and notebook.select() and notebook.tab(notebook.select(), "text") == "Statistics":
            canvas.resize(instant=True)

    def press_analysis_range(self, event):
        if event.button != 1 or event.xdata is None:
            return

        # Different event type
        mode = self.widgets['analysis_view']['toolbar'].mode
        if mode in (mode.ZOOM, mode.PAN):
            return

        # Determine which handle is dragged
        ax = self.widgets['analysis_view']['fig'].axes[0]
        start_px = ax.transData.transform((self.trace_information['range_start'].get(), 0))[0]
        end_px = ax.transData.transform((self.trace_information['range_end'].get(), 0))[0]
        if abs(event.x - start_px) < 10:
            self.dragged_range = '_range_start'
        elif abs(event.x - end_px) < 10:
            self.dragged_range = '_range_end'
        else:
            self.dragged_range = None
            return

        # Remove mpl lines
        for line in list(self.widgets['analysis_view']['fig'].axes[0].lines):
            if line.get_label() == self.dragged_range:
                line.remove()
        self.widgets['analysis_view']['canvas'].draw_idle()

        # Create temporary canvas line
        canvas = self.widgets['analysis_view']['canvas'].get_tk_widget()
        if self.range_line is not None:
            canvas.delete(self.range_line)
        self.range_line = canvas.create_line(
            event.x, (canvas.winfo_height() - ax.bbox.y1), event.x, (canvas.winfo_height() - ax.bbox.y0),
            fill=self.colours['baseline'],
            width=round(self.winfo_fpixels('1p') * 2),
        )

    def release_analysis_range(self, event):
        # Remove temporary canvas line
        canvas = self.widgets['analysis_view']['canvas'].get_tk_widget()
        if self.range_line is not None:
            self.after(50, lambda c = canvas, l = self.range_line: c.delete(l))
            self.range_line = None

        # Reset dragging state and do a proper update
        self.dragged_range = None
        self.update_analysis_range()

    def drag_analysis_range(self, event):
        if getattr(self, 'dragged_range', None) is None or event.xdata is None:
            return

        # Full range and user range
        x = float(event.xdata)
        full_start = float(self.time_vector[0])
        full_end = float(self.time_vector[len(self.time_vector) - 1])
        start_time = self.trace_information['range_start'].get()
        end_time = self.trace_information['range_end'].get()

        # Clamp it to the full range and the other handle
        if self.dragged_range == '_range_start':
            new_x = min(max(x, full_start), end_time)
            self.trace_information['range_start'].set(new_x)
        elif self.dragged_range == '_range_end':
            new_x = min(max(x, start_time), full_end)
            self.trace_information['range_end'].set(new_x)
        else:
            return

        # Move temporary canvas line only
        canvas = self.widgets['analysis_view']['canvas'].get_tk_widget()
        ax = self.widgets['analysis_view']['fig'].axes[0]
        if self.range_line is not None:
            x_px = ax.transData.transform((new_x, 0))[0]
            coords = canvas.coords(self.range_line)
            coords[0] = x_px
            coords[2] = x_px
            canvas.coords(self.range_line, *coords)

    def double_click_sash(self, event):
        paned = self.windows['paned']

        sash_index = None
        pane_index = None
        for i in range(len(paned.panes()) - 1):
            if abs(event.x - paned.sashpos(i)) <= _PADDING:
                sash_index = i
                pane_index = i if i == 0 else i + 1
                break

        if sash_index is None or pane_index is None:
            return

        pane = paned.nametowidget(paned.panes()[pane_index])
        current_pos = paned.sashpos(sash_index) if pane_index == 0 else  paned.winfo_width() - paned.sashpos(sash_index)
        if current_pos <= _PADDING:
            new_size = self.previous_pane_sizes.get(pane, max(self.width // 6, pane.winfo_reqwidth()))
        else:
            new_size = 0
            self.previous_pane_sizes[pane] = current_pos
        paned.sashpos(sash_index, new_size if pane_index == 0 else paned.winfo_width() - new_size)


    """
    Helpers
    """

    def downsample_line(self, x, y, xlim, max_points=None, baseline=None):
        if max_points is None:
            max_points = int(self.widgets['analysis_view']['canvas'].get_tk_widget().winfo_width())
            if max_points <= 1:
                max_points = int(self.width / 6 * 4)

        # Mask to xlim support
        mask1 = np.searchsorted(x, xlim[0], 'left')
        mask2 = np.searchsorted(x, xlim[1], 'right')
        x_visible = x[mask1:mask2]
        y_visible = y[mask1:mask2]

        # Baseline subtraction if enabled
        if self.figure_options['baseline_subtract'].get():
            current = self.analysis_results.get(self.last_called)
            if baseline is None and type(current.baseline) is np.ndarray:
                y_visible = y_visible - current.baseline[mask1:mask2]
            elif baseline is not None:
                y_visible = y_visible - baseline[mask1:mask2]

        # Embedded vectors have nan
        finite = np.isfinite(y_visible)
        if not finite.any():
            return np.array([], dtype=x_visible.dtype), np.array([], dtype=float)
        if not finite.all():
            x_visible = x_visible[finite]
            y_visible = y_visible[finite]

        length = len(x_visible)
        if length <= max_points:
            return x_visible, y_visible

        # Equally spaced bin edges
        starts, ends = _get_edges(length, max_points)

        # Per bin min and max
        y_min = np.minimum.reduceat(y_visible, starts)
        y_max = np.maximum.reduceat(y_visible, starts)

        # Allocate
        x_out = np.empty(2 * max_points, dtype=x_visible.dtype)
        y_out = np.empty(2 * max_points, dtype=y_visible.dtype)

        # Fill the output arrays
        x_out[0::2] = x_visible[starts]
        x_out[1::2] = x_visible[ends - 1]
        y_out[0::2] = y_min
        y_out[1::2] = y_max

        return x_out, y_out

    def downsample_events(self, xlim):
        # Total width budget
        width = int(self.widgets['analysis_view']['canvas'].get_tk_widget().winfo_width())
        if width <= 1:
            width = int(self.width / 6 * 4)

        # Save lookups
        current = self.analysis_results.get(self.last_called)
        coords = np.asarray(current.event_coordinates, dtype=int)
        if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
            return np.array([], dtype=object), np.array([], dtype=object)

        tv = self.time_vector
        tr = self.trace

        # Mask to xlim support
        visible = coords[(tv[coords[:, 1]] >= xlim[0]) & (tv[coords[:, 0]] <= xlim[1])]
        if len(visible) == 0:
            return np.array([], dtype=object), np.array([], dtype=object)

        # At least 2 pixels per segment
        per_seg = max(2, width // len(visible))

        # Downsample each visible event
        x_out, y_out = [], []
        for start, end in visible:
            # Grab and clamp each segment
            seg_x = tv[start:end + 1]
            seg_y = tr[start:end + 1]
            mask = (seg_x >= xlim[0]) & (seg_x <= xlim[1])
            if not mask.any():
                continue
            seg_x = seg_x[mask]
            seg_y = seg_y[mask]
            seg_baseline = current.baseline[start:end + 1][mask] if type(current.baseline) is np.ndarray else None

            # Downsample this segment with its own budget
            xs, ys = self.downsample_line(
                seg_x, seg_y,
                xlim=(seg_x[0], seg_x[len(seg_x) - 1]),
                max_points=per_seg,
                baseline=seg_baseline
            )

            x_out.extend(xs)
            x_out.append(np.nan)
            y_out.extend(ys)
            y_out.append(np.nan)

        return np.array(x_out, dtype=float), np.array(y_out, dtype=float)

    def downsample_all(self):
        ax = self.widgets['analysis_view']['fig'].axes[0]
        current = self.analysis_results.get(self.last_called)
        xlim = ax.get_xlim()
        thresh = 0
        for line in ax.get_lines():
            if line.get_label() not in self.colours.keys():
                continue
            if line.get_label() == 'threshold':
                if thresh == 0:
                    line.set_data(*self.downsample_line(self.time_vector, current.pos_threshold, xlim))
                else:
                    line.set_data(*self.downsample_line(self.time_vector, current.neg_threshold, xlim))
                thresh += 1
            elif line.get_label() == 'events':
                line.set_data(*self.downsample_events(xlim))
            else:
                vector = getattr(current, line.get_label(), None)
                if vector is None:
                    vector = getattr(self, line.get_label())
                line.set_data(*self.downsample_line(self.time_vector, vector, xlim))

    def downsample_refresh(self):
        self.downsample_all()
        self.widgets['analysis_view']['fig'].canvas.draw_idle()
        self.widgets['analysis_view']['fig'].canvas.flush_events()

    def embed_vector(self, vector, start, stop):
        out = np.full(len(self.trace), np.nan, dtype=float)
        out[start:stop] = vector
        return out


    """
    Algorithm processing 
    """

    def process_results(self, results):
        # Get previous analysis range
        last_range = getattr(self.analysis_results.get(self.last_called), "analysis_range", None)

        # Get the final iteration
        self.last_called = 'Iterate' if isinstance(results, IterateResults) else 'Estimate'
        self.analysis_results[self.last_called] = results.iterations[len(results.iterations) - 1]
        current = self.analysis_results[self.last_called]

        # Range and cutoff used
        current.analysis_range = self.pending_range or (0, len(self.trace))
        current.cutoff = results.args.cutoff

        # Embed all output vectors
        current.filtered_trace = self.embed_vector(results.args.filtered_trace, *current.analysis_range)
        current.initial_threshold = self.embed_vector(results.initial_threshold, *current.analysis_range)
        current.calculation_trace = self.embed_vector(current.calculation_trace, *current.analysis_range)
        current.baseline = self.embed_vector(current.baseline, *current.analysis_range)
        thresh = self.embed_vector(current.threshold, *current.analysis_range)
        current.pos_threshold = current.baseline + thresh
        current.neg_threshold = current.baseline - thresh

        # Embed events
        current.events = results.events
        current.events.label = self.last_called
        current.event_coordinates += current.analysis_range[0]
        for event in current.events:
            coords = event['Coordinates']
            event['Coordinates'] = (coords[0] + current.analysis_range[0], coords[1] + current.analysis_range[0])

        # Get statistics
        if isinstance(results, IterateResults):
            self.analysis_results['Statistics'] = dict(
                trace_stats=[res.trace_stats for res in results.iterations],
                event_stats=[res.event_stats for res in results.iterations])
        else:
            current.trace_stats = results.iterations[0].trace_stats
            current.event_direction = results.event_direction

        # Update the results window
        self.update_results()
        # Show the vectors (relim if detrended and range is different)
        retain_view = True
        if self.figure_options['baseline_subtract'].get() is True:
            if last_range is not None and last_range != current.analysis_range:
                retain_view = False
        self.update_figure(title=self.last_called, retain_view=retain_view)

        # Update the direction for estimate only
        if hasattr(current, 'event_direction'):
            self.analysis_options['event_direction'].set(current.event_direction)
            self.flash_entry(self.widgets['analyse']['event_direction_input'])
            self.flash_entry(self.widgets['analyse']['iterate'])

        # Update the cutoff
        self.analysis_options['cutoff'].set(np.round(current.event_stats['Max Cutoff'], 2))
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
                units = getattr(data, "sweepUnitsY", "pA")
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
                    units = 'pA'
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
        # Update the trace information
        self.trace_information['units'].set(units)
        # Show the trace information and trace vector
        self.update_load(path)
        # Update the figure
        self.update_figure(title='Trace', retain_view=False)
        # Set the z-score
        self.update_z_score()
        # Set the flag
        self.flags['loaded'] = True
        # Update options
        self.update_options()

    def landing_load(self):
        self.load_trace()

        if self.flags['loaded']:
            self.toggle_landing_page(False)

    def check_thread(self, thread, results, errors, current_time, title):
        self.flags['running'] = True

        # Check for errors
        if not errors.empty():
            self.widgets['analysis_view']['progress_bar'].stop()
            self.flags['running'] = False
            messagebox.showerror('Error', f'{title} failed: {errors.get()}')
            return

        # Reschedule check
        if thread.is_alive():
            self.after(100, lambda: self.check_thread(thread, results, errors, current_time, title))
            return

        # Get results
        thread.join()
        self.update_idletasks()
        try:
            results = results.get_nowait()
        except queue.Empty:
            self.widgets['analysis_view']['progress_bar'].stop()
            self.flags['running'] = False
            messagebox.showerror('Error', f'{title} failed: No results returned')
            return

        # Update analysis results
        print(f'{title} time: {time.time() - current_time:.4f}')
        self.widgets['analysis_view']['progress_bar'].stop()
        self.flags['running'] = False
        self.process_results(results)
        # Set the focus and flag
        if title == 'Estimate':
            self.widgets['analyse']['iterate'].focus_set()
            self.flags['estimated'] = True
        else:
            self.widgets['results']['save'].focus_set()
            self.flags['iterated'] = True
        # Update options
        self.update_options()

    def initial_estimate(self):
        if not self.flags['loaded']:
            messagebox.showerror('Error', 'Trace not loaded')
            return
        if self.flags['running']:
            messagebox.showerror('Error', 'Work in progress')
            return

        current_time = time.time()
        self.widgets['analysis_view']['progress_bar'].start(10)

        # Get current analysis range
        start_idx, end_idx = self.current_range or (0, len(self.trace))
        self.pending_range = (start_idx, end_idx)
        trace = self.trace[start_idx:end_idx]
        # Filter the trace
        filtered_trace = nn.methods._bounds_filter(trace, self.analysis_options['_bounds_filter'].get())

        # Estimate the cutoff and direction
        def worker(results, errors, *args):
            try:
                results.put(nn.initial_estimate(*args))
            except Exception as e:
                errors.put(str(e))
                return

        # Run on a separate thread
        results = queue.Queue()
        errors = queue.Queue()
        thread = Thread(
            target=worker,
            args=(
                results,
                errors,
                trace,
                filtered_trace,
                int(1 / self.time_step),
                self.analysis_options['estimate_cutoff'].get(),
                self.analysis_options['replace_factor'].get(),
                self.analysis_options['replace_gap'].get(),
                self.analysis_options['threshold_window'].get(),
                self.analysis_options['z_score'].get(),
                self.analysis_options['features'].get(),
                False,
                self.analysis_options['parallel'].get()
            ),
            daemon=True
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

        # Get current analysis range
        start_idx, end_idx = self.current_range or (0, len(self.trace))
        self.pending_range = (start_idx, end_idx)
        trace = self.trace[start_idx:end_idx]
        # Filter the trace
        filtered_trace = nn.methods._bounds_filter(trace, self.analysis_options['_bounds_filter'].get())

        # Iterate to find the final baseline and threshold
        def worker(results, errors, *args):
            try:
                results.put(nn.iterate(*args))
            except Exception as e:
                errors.put(str(e))
                return

        # Run on a separate thread
        results = queue.Queue()
        errors = queue.Queue()
        thread = Thread(
            target=worker,
            args=(
                results,
                errors,
                trace,
                self.analysis_options['cutoff'].get(),
                self.analysis_options['event_direction'].get(),
                filtered_trace,
                int(1 / self.time_step),
                self.analysis_options['replace_factor'].get(),
                self.analysis_options['replace_gap'].get(),
                self.analysis_options['threshold_window'].get(),
                self.analysis_options['z_score'].get(),
                self.analysis_options['features'].get(),
                False,
                self.analysis_options['parallel'].get()
            ),
            daemon=True
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
        results = self.analysis_results.get(self.last_called)
        if not results:
            messagebox.showerror('Error', 'No results to save')
            return
        results = copy.deepcopy(results)
        event_vectors = np.asarray([event['Vector'] for event in results.events], dtype=object)

        # Save the results
        path = tk.filedialog.asksaveasfilename(
            initialfile=basename(self.trace_information['path'].get()).split('.')[0] + f'_{self.last_called}_events.csv',
            defaultextension=".csv",
            initialdir=os.path.split(self.trace_information['path'].get())[0],
            title="Choose a location for the events",
            filetypes=([("CSV files", "*.csv")])
        )
        if path not in ['', None]:
            # If you don't do this, DictWriter will ruin your life :( (** 2 bcos of excel cell size limit)
            units = self.trace_information['units'].get()
            amp_key = 'Amplitude (A)' if units == 'A' else f'Amplitude ({units[0]}A)'
            area_key = 'Area (C)' if units == 'A' else f'Area ({units[0]}C)'
            for event in results.events:
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

            with open(path, 'w', encoding='utf8', newline='') as file:
                w = csv.DictWriter(file, fieldnames)
                w.writeheader()
                w.writerows(results.events)

            # Success
            messagebox.showinfo('Success', f'Saved {self.last_called} events')

            # Set the flag
            self.flags['saved'] = True

        # Save arrays
        array_path = tk.filedialog.asksaveasfilename(
            initialfile=basename(self.trace_information['path'].get()).split('.')[0] + f'_{self.last_called}_arrays.npz',
            defaultextension='.npz',
            initialdir=os.path.split(self.trace_information['path'].get())[0],
            title='Choose a location for the arrays',
            filetypes=([('NumPy compressed archive', '*.npz')])
        )
        if array_path not in ['', None]:
            array_names = [
                'trace',
                'time_vector',
                'time_step',
                'filtered_trace',
                'baseline',
                'pos_threshold',
                'initial_threshold',
                'calculation_trace',
                'event_coordinates',
            ]
            arrays = {}
            for name in array_names:
                array = getattr(self, name, None)
                if array is None:
                    array = getattr(results, name)
                array = np.asarray(array)
                if name == 'pos_threshold':
                    name = 'threshold'
                    array = array - results.baseline
                arrays[name] = array
            arrays['event_vectors'] = event_vectors
            np.savez_compressed(array_path, **arrays)

            # Success
            messagebox.showinfo('Success', f'Saved {self.last_called} arrays')

            # Set the flag
            self.flags['saved'] = True
