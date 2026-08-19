#!/usr/bin/env python3
"""
Log Analysis Script for Robot Navigation Data

This script analyzes CSV log files containing robot navigation data and generates
visualizations for time to goal, distance to goal, and alignment with goal.

CSV Format:
- Column 0: Starting step of current path
- Column 1: Current step  
- Column 2: Distance to path
- Column 3: Angle distance to path (alignment)
- Column 4: Distance to goal

Simulation step: 20ms
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple

# "Goal reached" thresholds: reuse the exact same constants the env uses to define
# a reached goal pose, so the analysis and the eval agree. Fall back to the env's
# defaults if the package can't be imported (e.g. running the script standalone).
try:
    from mjlab_piplus.tasks.piplus_path_env_cfg import (
        GOAL_POS_SUCCESS,
        GOAL_YAW_SUCCESS,
    )
except Exception:  # pragma: no cover - import is best-effort
    GOAL_POS_SUCCESS = 0.1  # meters (footprint distance to the goal)
    GOAL_YAW_SUCCESS = 0.2  # radians (heading error to the goal yaw)

# Use matplotlib default style
plt.style.use('default')

class LogAnalyzer:
    def __init__(self, simulation_step_ms: float = 20.0, folder_names: Dict[str, str] = None,
                 command_timeout_s: float = 30.0):
        """
        Initialize the log analyzer.

        Args:
            simulation_step_ms: Time per simulation step in milliseconds
            folder_names: Optional dictionary mapping folder paths to display names
            command_timeout_s: The command's resampling timeout (seconds). A log
                that doesn't reach the goal but ran this long is a "timeout"; a
                shorter one is a "fail" (fell over / cut off).
        """
        self.simulation_step_ms = simulation_step_ms
        self.folder_names = folder_names or {}
        self.command_timeout_s = command_timeout_s
        self.results = {}
        
    def find_csv_files(self, folder_path: str) -> List[str]:
        """
        Find all CSV files with 3-digit numbers in the given folder.
        
        Args:
            folder_path: Path to the folder to search
            
        Returns:
            List of CSV file paths
        """
        pattern = os.path.join(folder_path, "*[0-9][0-9][0-9].csv")
        csv_files = glob.glob(pattern)
        
        # Also check for path_logs pattern specifically
        path_logs_pattern = os.path.join(folder_path, "path_logs[0-9][0-9][0-9].csv")
        csv_files.extend(glob.glob(path_logs_pattern))
        
        # Remove duplicates and sort
        csv_files = sorted(list(set(csv_files)))
        
        print(f"Found {len(csv_files)} CSV files in {folder_path}")
        return csv_files
    
    def read_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file and return a DataFrame with proper column names.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the log data
        """
        try:
            # Read CSV without headers
            df = pd.read_csv(file_path, header=None)
            
            # Assign column names based on the specification. The piplus logs add
            # a 6th column (yaw error to the goal orientation) that the x02 logs
            # did not have.
            df.columns = ['starting_step', 'current_step', 'distance_to_path',
                         'angle_distance_to_path', 'distance_to_goal',
                         'yaw_error_to_goal']
            
            return df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def calculate_metrics(self, df: pd.DataFrame, file_path: str) -> Dict:
        """
        Calculate metrics for a single CSV file.
        
        Args:
            df: DataFrame with the log data
            file_path: Path to the CSV file for identification
            
        Returns:
            Dictionary with calculated metrics
        """
        if df is None or df.empty:
            return None
            
        # Calculate time to goal
        total_steps = len(df)
        time_to_goal_ms = total_steps * self.simulation_step_ms
        time_to_goal_s = time_to_goal_ms / 1000.0
        
        # Calculate average distance to goal
        avg_distance_to_goal = df['distance_to_goal'].mean()
        
        # Calculate average alignment (inverse of angle distance)
        # Smaller angle distance means better alignment
        avg_angle_distance = df['angle_distance_to_path'].mean()
        
        # Final distance and yaw error to the goal (goal reached?). Matches the
        # env's goal-pose criterion: footprint within GOAL_POS_SUCCESS AND heading
        # within GOAL_YAW_SUCCESS of the goal yaw.
        final_distance_to_goal = df['distance_to_goal'].iloc[-1]
        final_yaw_error_to_goal = df['yaw_error_to_goal'].iloc[-1]

        # Calculate average distance to path
        avg_distance_to_path = df['distance_to_path'].mean()

        # Goal reached = full pose (position AND orientation), using the env
        # constants. Then classify the outcome: success / timeout / fail.
        goal_reached = bool(
            (final_distance_to_goal < GOAL_POS_SUCCESS)
            and (final_yaw_error_to_goal < GOAL_YAW_SUCCESS)
        )
        if goal_reached:
            classification = 'success'
        elif time_to_goal_s >= self.command_timeout_s - (self.simulation_step_ms / 1000.0):
            # Ran (essentially) the full command timeout without reaching the goal.
            classification = 'timeout'
        else:
            # Ended before the timeout without reaching the goal (fell over, or the
            # eval run was cut off mid-path).
            classification = 'fail'

        # Extract file number for identification
        file_name = os.path.basename(file_path)
        file_number_match = re.search(r'(\d{3})', file_name)
        file_number = int(file_number_match.group(1)) if file_number_match else 0
        
        return {
            'file_path': file_path,
            'file_name': file_name,
            'file_number': file_number,
            'total_steps': total_steps,
            'time_to_goal_ms': time_to_goal_ms,
            'time_to_goal_s': time_to_goal_s,
            'avg_distance_to_goal': avg_distance_to_goal,
            'avg_angle_distance': avg_angle_distance,
            'avg_distance_to_path': avg_distance_to_path,
            'final_distance_to_goal': final_distance_to_goal,
            'final_yaw_error_to_goal': final_yaw_error_to_goal,
            # Full goal pose: position AND orientation, using the env's constants.
            'goal_reached': goal_reached,
            # Outcome class: a log that reached the goal is a "success"; one that
            # didn't but ran the full command timeout is a "timeout"; a shorter one
            # ended early (fell over / run cut off) and is a "fail". Only successes
            # have a meaningful time-to-goal.
            'classification': classification,
        }
    
    def analyze_folder(self, folder_path: str) -> Dict:
        """
        Analyze all CSV files in a folder.
        
        Args:
            folder_path: Path to the folder to analyze
            
        Returns:
            Dictionary with analysis results for the folder
        """
        csv_files = self.find_csv_files(folder_path)
        
        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            return None
            
        folder_results = {
            'folder_path': folder_path,
            'folder_name': os.path.basename(folder_path) if folder_path != '.' else 'root',
            'files': [],
            'metrics': []
        }
        
        for csv_file in csv_files:
            df = self.read_csv_file(csv_file)
            metrics = self.calculate_metrics(df, csv_file)
            
            if metrics:
                folder_results['files'].append(csv_file)
                folder_results['metrics'].append(metrics)
        
        print(f"Analyzed {len(folder_results['metrics'])} files in {folder_path}")
        return folder_results
    
    def analyze_folders(self, folder_paths: List[str]) -> None:
        """
        Analyze multiple folders and store results.
        
        Args:
            folder_paths: List of folder paths to analyze
        """
        self.results = {}
        
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist")
                continue
                
            folder_results = self.analyze_folder(folder_path)
            if folder_results:
                # Use custom name if provided, otherwise use default folder name
                display_name = self.folder_names.get(folder_path, folder_results['folder_name'])
                folder_results['folder_name'] = display_name
                self.results[folder_path] = folder_results
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame from all analyzed results.
        
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for folder_path, folder_results in self.results.items():
            for metrics in folder_results['metrics']:
                row = {
                    'folder': folder_results['folder_name'],
                    'file_number': metrics['file_number'],
                    'time_to_goal_s': metrics['time_to_goal_s'],
                    'avg_distance_to_goal': metrics['avg_distance_to_goal'],
                    'avg_angle_distance': metrics['avg_angle_distance'],
                    'avg_distance_to_path': metrics['avg_distance_to_path'],
                    'final_distance_to_goal': metrics['final_distance_to_goal'],
                    'goal_reached': metrics['goal_reached'],
                    'classification': metrics['classification'],
                    'total_steps': metrics['total_steps']
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def plot_results(self, save_plots: bool = True, output_dir: str = 'plots') -> None:
        """
        Create comprehensive plots of the analysis results as separate individual plots.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        if not self.results:
            print("No results to plot. Run analyze_folders() first.")
            return
        
        # Create output directory if saving plots
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create summary dataframe
        summary_df = self.create_summary_dataframe()
        
        if summary_df.empty:
            print("No data to plot")
            return
        
        plt.rcParams.update({
            "text.usetex": True,           # Use LaTeX to render all text
            "font.family": "serif",        # Use serif fonts
            "font.serif": ["Times"],       # Match IEEE Times New Roman style
            "pdf.fonttype": 42,            # Output Type 42 (TrueType) fonts
            "ps.fonttype": 42,             # Output Type 42 (TrueType) fonts
            "font.size": 20                 # Match typical caption/label sizes
        })
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        # Extend colors if we need more than the default cycle
        if len(summary_df['folder'].unique()) > len(colors):
            colors = colors * ((len(summary_df['folder'].unique()) // len(colors)) + 1)
        
        # Plot 1: Time to Goal by Folder (Horizontal Violin Plot).
        # Time-to-goal is only meaningful for runs that reached the goal, so this
        # distribution uses successful runs only.
        success_df = summary_df[summary_df['classification'] == 'success']
        plt.figure(figsize=(10, 6))
        if len(summary_df['folder'].unique()) > 1:
            # Prepare data for violin plot: one entry per folder that has at least
            # one successful run (violinplot can't take an empty dataset).
            data_for_violin = []
            plot_positions = []
            plot_labels = []
            for folder in summary_df['folder'].unique():
                folder_data = success_df[success_df['folder'] == folder]['time_to_goal_s'].values
                if len(folder_data) == 0:
                    continue
                data_for_violin.append(folder_data)
                plot_positions.append(len(plot_positions) * 0.7 + 1)
                plot_labels.append(folder)

            # Create horizontal violin plot with custom positions
            parts = plt.violinplot(data_for_violin, positions=plot_positions, vert=False, showmeans=True, showmedians=True)

            # Customize violin plot colors
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)

            # Set y-axis labels to folder names with custom positions
            plt.yticks(plot_positions, plot_labels)
            plt.xlabel('Time to Goal (seconds)')
            #plt.ylabel('Folder')
            plt.tight_layout()
            #plt.title('Time to Goal Distribution (successful runs only)')
            plt.grid(True, alpha=0.3, axis='x')
        else:
            # For single folder, still use histogram
            plt.hist(success_df['time_to_goal_s'], bins=20, alpha=0.7, color=colors[0], orientation='horizontal')
            #plt.title('Time to Goal Distribution (successful runs only)')
            plt.xlabel('Frequency')
            plt.ylabel('Time to Goal (seconds)')
            plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '1_time_to_goal.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Average Distance to Goal by Folder (Horizontal Violin Plot)
        plt.figure(figsize=(10, 6))
        if len(summary_df['folder'].unique()) > 1:
            # Prepare data for violin plot
            folder_names = summary_df['folder'].unique()
            data_for_violin = []
            for folder in folder_names:
                folder_data = summary_df[summary_df['folder'] == folder]['avg_distance_to_goal'].values
                data_for_violin.append(folder_data)
            
            # Create custom positions with reduced spacing
            positions = [i * 0.7 + 1 for i in range(len(folder_names))]
            
            # Create horizontal violin plot with custom positions
            parts = plt.violinplot(data_for_violin, positions=positions, vert=False, showmeans=True, showmedians=True)
            
            # Customize violin plot colors
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            
            # Set y-axis labels to folder names with custom positions
            plt.yticks(positions, folder_names)
            plt.xlabel('Mean Distance to Goal')
            #plt.ylabel('Folder')
            plt.title('Mean Distance to Goal Distribution')
            plt.grid(True, alpha=0.3, axis='x')
        else:
            # For single folder, still use histogram
            plt.hist(summary_df['avg_distance_to_goal'], bins=20, alpha=0.7, color=colors[0], orientation='horizontal')
            plt.title('Mean Distance to Goal Distribution')
            plt.xlabel('Frequency')
            plt.ylabel('Mean Distance to Goal')
            plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '2_avg_distance_to_goal.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Mean Angle Distance (Alignment) by Folder (Horizontal Violin Plot)
        plt.figure(figsize=(10, 6))
        if len(summary_df['folder'].unique()) > 1:
            # Prepare data for violin plot
            folder_names = summary_df['folder'].unique()
            data_for_violin = []
            for folder in folder_names:
                folder_data = summary_df[summary_df['folder'] == folder]['avg_angle_distance'].values
                data_for_violin.append(folder_data)
            
            # Create custom positions with reduced spacing
            positions = [i * 0.7 + 1 for i in range(len(folder_names))]
            
            # Create horizontal violin plot with custom positions
            parts = plt.violinplot(data_for_violin, positions=positions, vert=False, showmeans=True, showmedians=True)
            
            # Customize violin plot colors
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            
            # Set y-axis labels to folder names with custom positions
            plt.yticks(positions, folder_names)
            plt.xlabel('Mean Angle Distance (radians)')
            #plt.ylabel('Folder')
            plt.title('Mean Angle Distance Distribution')
            plt.grid(True, alpha=0.3, axis='x')
        else:
            # For single folder, still use histogram
            plt.hist(summary_df['avg_angle_distance'], bins=20, alpha=0.7, color=colors[0], orientation='horizontal')
            plt.title('Mean Angle Distance Distribution')
            plt.xlabel('Frequency')
            plt.ylabel('Mean Angle Distance to Path Direction(radians)')
            plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '3_avg_angle_distance.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 4: mean Distance to Path by Folder (Horizontal Violin Plot).
        # Successful runs only, to match the time-to-goal distribution.
        plt.figure(figsize=(10, 6))
        if len(summary_df['folder'].unique()) > 1:
            # Prepare data for violin plot: one entry per folder that has at least
            # one successful run (violinplot can't take an empty dataset).
            data_for_violin = []
            plot_positions = []
            plot_labels = []
            for folder in summary_df['folder'].unique():
                folder_data = success_df[success_df['folder'] == folder]['avg_distance_to_path'].values
                if len(folder_data) == 0:
                    continue
                data_for_violin.append(folder_data)
                plot_positions.append(len(plot_positions) * 0.7 + 1)
                plot_labels.append(folder)

            # Create horizontal violin plot with custom positions
            parts = plt.violinplot(data_for_violin, positions=plot_positions, vert=False, showmeans=True, showmedians=True)

            # Customize violin plot colors
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)

            # Set y-axis labels to folder names with custom positions
            plt.yticks(plot_positions, plot_labels)
            plt.xlabel('Mean Distance to Path (m)')
            #plt.ylabel('Folder')
            #plt.title('Mean Distance to Path Distribution (successful runs only)')
            plt.grid(True, alpha=0.3, axis='x')
        else:
            # For single folder, still use histogram
            plt.hist(success_df['avg_distance_to_path'], bins=20, alpha=0.7, color=colors[0], orientation='horizontal')
            #plt.title('Mean Distance to Path Distribution (successful runs only)')
            plt.xlabel('Frequency')
            plt.ylabel('Mean Distance to Path (m)')
            plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '4_avg_distance_to_path.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 5: Correlation between Time and Distance to Goal
        plt.figure(figsize=(10, 6))
        for i, folder in enumerate(summary_df['folder'].unique()):
            folder_data = summary_df[summary_df['folder'] == folder]
            plt.scatter(folder_data['time_to_goal_s'], folder_data['avg_distance_to_goal'], 
                       label=folder, alpha=0.7, color=colors[i], s=50)
        plt.xlabel('Time to Goal (seconds)')
        plt.ylabel('Average Distance to Goal')
        plt.title('Time vs Distance to Goal Correlation')
        if len(summary_df['folder'].unique()) > 1:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '5_time_vs_distance_correlation.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 6: Goal Reached Success Rate
        plt.figure(figsize=(10, 6))
        if len(summary_df['folder'].unique()) > 1:
            # Keep the folder order (and thus the color-by-position) consistent with
            # every other plot, which uses insertion order. groupby() would sort the
            # folders alphabetically and mis-align the colors.
            folder_order = summary_df['folder'].unique()
            success_rates = summary_df.groupby('folder')['goal_reached'].mean().reindex(folder_order)
            ax = success_rates.plot(kind='bar', color=[colors[i % len(colors)] for i in range(len(success_rates))])
            #plt.title('Goal Reached Success Rate')
            plt.ylabel('Success Rate')
            plt.xlabel("")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            # Add percentage labels on bars
            for i, v in enumerate(success_rates.values):
                ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        else:
            overall_success_rate = summary_df['goal_reached'].mean()
            plt.bar(['Overall'], [overall_success_rate], color=colors[0])
            plt.title('Overall Goal Reached Success Rate')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.text(0, overall_success_rate + 0.02, f'{overall_success_rate:.1%}', ha='center', va='bottom')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '6_success_rate.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 7: Distance to Path vs Distance to Goal
        plt.figure(figsize=(10, 6))
        for i, folder in enumerate(summary_df['folder'].unique()):
            folder_data = summary_df[summary_df['folder'] == folder]
            plt.scatter(folder_data['avg_distance_to_path'], folder_data['avg_distance_to_goal'], 
                       label=folder, alpha=0.7, color=colors[i], s=50)
        plt.xlabel('Average Distance to Path')
        plt.ylabel('Average Distance to Goal')
        plt.title('Path Following vs Goal Reaching Performance')
        if len(summary_df['folder'].unique()) > 1:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '7_path_vs_goal_distance.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 8: Time Series of File Numbers (Learning Progress)
        plt.figure(figsize=(12, 6))
        for i, folder in enumerate(summary_df['folder'].unique()):
            folder_data = summary_df[summary_df['folder'] == folder].sort_values('file_number')
            plt.plot(folder_data['file_number'], folder_data['time_to_goal_s'], 
                    label=folder, marker='o', alpha=0.7, color=colors[i], markersize=4)
        plt.xlabel('Episode Number')
        plt.ylabel('Time to Goal (seconds)')
        plt.title('Learning Progress: Time to Goal Over Episodes')
        if len(summary_df['folder'].unique()) > 1:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '8_learning_progress.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 9: Final Distance to Goal Distribution
        plt.figure(figsize=(10, 6))
        if len(summary_df['folder'].unique()) > 1:
            summary_df.boxplot(column='final_distance_to_goal', by='folder', ax=plt.gca())
            plt.title('Final Distance to Goal by Folder')
            plt.suptitle('')  # Remove automatic title from boxplot
        else:
            plt.hist(summary_df['final_distance_to_goal'], bins=20, alpha=0.7, color=colors[0])
            plt.title('Final Distance to Goal Distribution')
            plt.xlabel('Final Distance to Goal')
            plt.ylabel('Frequency')
        plt.ylabel('Final Distance to Goal')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, '9_final_distance_to_goal.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 10: Summary Statistics Text Report
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        stats_summary = summary_df.groupby('folder').agg({
            'time_to_goal_s': ['mean', 'std', 'min', 'max'],
            'avg_distance_to_goal': ['mean', 'std'],
            'avg_angle_distance': ['mean', 'std'],
            'goal_reached': 'mean'
        }).round(3)
        
        # Create a comprehensive text summary
        summary_text = "COMPREHENSIVE ANALYSIS SUMMARY\n"
        summary_text += "="*50 + "\n\n"
        
        for folder in stats_summary.index:
            summary_text += f"{folder.upper()}\n"
            summary_text += "-" * 30 + "\n"
            summary_text += f"Performance Metrics:\n"
            summary_text += f"  ⏱️  Average time to goal: {stats_summary.loc[folder, ('time_to_goal_s', 'mean')]:.2f} ± {stats_summary.loc[folder, ('time_to_goal_s', 'std')]:.2f} seconds\n"
            summary_text += f"  ⚡ Best time to goal: {stats_summary.loc[folder, ('time_to_goal_s', 'min')]:.2f} seconds\n"
            summary_text += f"  🐌 Worst time to goal: {stats_summary.loc[folder, ('time_to_goal_s', 'max')]:.2f} seconds\n"
            summary_text += f"  🎯 Average distance to goal: {stats_summary.loc[folder, ('avg_distance_to_goal', 'mean')]:.3f} ± {stats_summary.loc[folder, ('avg_distance_to_goal', 'std')]:.3f}\n"
            summary_text += f"  🧭 Average angle distance: {stats_summary.loc[folder, ('avg_angle_distance', 'mean')]:.3f} ± {stats_summary.loc[folder, ('avg_angle_distance', 'std')]:.3f}\n"
            summary_text += f"  ✅ Goal success rate: {stats_summary.loc[folder, ('goal_reached', 'mean')]:.1%}\n"
            
            # Add episode counts
            folder_data = summary_df[summary_df['folder'] == folder]
            summary_text += f"  📊 Total episodes analyzed: {len(folder_data)}\n"
            summary_text += f"  🏆 Successful episodes: {folder_data['goal_reached'].sum()}\n\n"
        
        #plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
        #        fontsize=11, verticalalignment='top', fontfamily='monospace',
        #        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        #plt.title('Statistical Summary Report', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        #if save_plots:
        #    plt.savefig(os.path.join(output_dir, '10_summary_statistics.pdf'), dpi=300, bbox_inches='tight')
        #plt.show()
        
        print(f"\n✅ Generated 10 individual analysis plots")
        if save_plots:
            print(f"📁 All plots saved to: {output_dir}/")
        
        # Create individual detailed plots for each folder
        self._create_detailed_plots(summary_df, save_plots, output_dir)
    
    def _create_detailed_plots(self, summary_df: pd.DataFrame, save_plots: bool, output_dir: str):
        """Create detailed plots for each folder."""
        
        for folder_name in summary_df['folder'].unique():
            folder_data = summary_df[summary_df['folder'] == folder_name]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Detailed Analysis: {folder_name}', fontsize=16)
            
            # Time progression
            sorted_data = folder_data.sort_values('file_number')
            ax1.plot(sorted_data['file_number'], sorted_data['time_to_goal_s'], 'o-')
            ax1.set_xlabel('Episode Number')
            ax1.set_ylabel('Time to Goal (seconds)')
            ax1.set_title('Learning Progress: Time to Goal')
            ax1.grid(True, alpha=0.3)
            
            # Distance metrics
            ax2.plot(sorted_data['file_number'], sorted_data['avg_distance_to_goal'], 'o-', label='Distance to Goal')
            ax2.plot(sorted_data['file_number'], sorted_data['avg_distance_to_path'], 'o-', label='Distance to Path')
            ax2.set_xlabel('Episode Number')
            ax2.set_ylabel('Distance')
            ax2.set_title('Distance Metrics Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Angle distance (alignment)
            ax3.plot(sorted_data['file_number'], sorted_data['avg_angle_distance'], 'o-', color='red')
            ax3.set_xlabel('Episode Number')
            ax3.set_ylabel('Average Angle Distance')
            ax3.set_title('Alignment with Goal Direction')
            ax3.grid(True, alpha=0.3)
            
            # Success indicators
            success_data = sorted_data['goal_reached'].astype(int)
            ax4.plot(sorted_data['file_number'], success_data, 'o-', color='green')
            ax4.set_xlabel('Episode Number')
            ax4.set_ylabel('Goal Reached (1=Yes, 0=No)')
            ax4.set_title('Goal Achievement Over Time')
            ax4.set_ylim(-0.1, 1.1)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = os.path.join(output_dir, f'detailed_analysis_{folder_name}.pdf')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Detailed plot saved for {folder_name}: {plot_path}")
            
            plt.show()
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        if not self.results:
            print("No results available. Run analyze_folders() first.")
            return
        
        summary_df = self.create_summary_dataframe()
        
        print("\n" + "="*60)
        print("LOG ANALYSIS SUMMARY")
        print("="*60)
        
        for folder_name in summary_df['folder'].unique():
            folder_data = summary_df[summary_df['folder'] == folder_name]
            n = len(folder_data)

            # Outcome breakdown.
            n_success = int((folder_data['classification'] == 'success').sum())
            n_timeout = int((folder_data['classification'] == 'timeout').sum())
            n_fail = int((folder_data['classification'] == 'fail').sum())

            # Performance metrics are only meaningful for successful paths — a
            # timed-out / failed path has no valid time-to-goal — so average them
            # over successes only.
            succ = folder_data[folder_data['classification'] == 'success']

            print(f"\nFolder: {folder_name}")
            print("-" * 40)
            print(f"Number of files: {n}")
            print(f"Outcomes: {n_success} success ({n_success / n:.1%}), "
                  f"{n_timeout} timeout ({n_timeout / n:.1%}), "
                  f"{n_fail} fail ({n_fail / n:.1%})")
            if n_success == 0:
                print("No successful paths — no time-to-goal statistics.")
                continue
            print("  (stats below over successful paths only)")
            print(f"Average time to goal: {succ['time_to_goal_s'].mean():.2f} ± {succ['time_to_goal_s'].std():.2f} seconds")
            print(f"Average distance to goal: {succ['avg_distance_to_goal'].mean():.3f} ± {succ['avg_distance_to_goal'].std():.3f}")
            print(f"Average angle distance: {succ['avg_angle_distance'].mean():.3f} ± {succ['avg_angle_distance'].std():.3f}")
            print(f"Average distance to path: {succ['avg_distance_to_path'].mean():.3f} ± {succ['avg_distance_to_path'].std():.3f}")
            print(f"Min time to goal: {succ['time_to_goal_s'].min():.2f} seconds")
            print(f"Max time to goal: {succ['time_to_goal_s'].max():.2f} seconds")


def main():
    """Main function to run the log analysis."""
    parser = argparse.ArgumentParser(description='Analyze robot navigation log files')
    parser.add_argument('folders', nargs='+', help='Folders containing CSV log files')
    parser.add_argument('--simulation-step', type=float, default=20.0,
                       help='Simulation step in milliseconds (default: 20.0)')
    parser.add_argument('--command-timeout', type=float, default=30.0,
                       help='Command resampling timeout in seconds; a non-reaching '
                            'log this long is a timeout, shorter is a fail (default: 30.0)')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save plots to files')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    parser.add_argument('--folder-names', type=str, nargs='*',
                       help='Custom names for folders (same order as folders argument). '
                            'Example: --folder-names "Baseline" "Improved" "Final"')
    
    args = parser.parse_args()
    
    # Create folder names mapping if provided
    folder_names = {}
    if args.folder_names:
        if len(args.folder_names) != len(args.folders):
            print(f"Error: Number of folder names ({len(args.folder_names)}) must match number of folders ({len(args.folders)})")
            return
        folder_names = dict(zip(args.folders, args.folder_names))
        print(f"Using custom folder names: {folder_names}")
    
    # Initialize analyzer
    analyzer = LogAnalyzer(simulation_step_ms=args.simulation_step, folder_names=folder_names,
                           command_timeout_s=args.command_timeout)
    
    # Analyze folders
    print(f"Analyzing folders: {args.folders}")
    analyzer.analyze_folders(args.folders)
    
    # Print summary
    analyzer.print_summary()
    
    # Create plots
    analyzer.plot_results(save_plots=not args.no_save, output_dir=args.output_dir)


if __name__ == "__main__":
    main()