#!/usr/bin/env python3
"""
Interactive Plotly visualization for velocity initialization.
Creates HTML files with:
1. 3D point cloud with velocity heatmap
2. Animation showing motion over time
3. Interactive time slider
"""

import sys
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))

from datasets.FreeTime_dataset import (
    FreeTimeParser,
    load_multiframe_colmap_grid_tracked,
)


def visualize_velocity_plotly(
    data_dir: str,
    output_dir: str = "velocity_debug",
    start_frame: int = 0,
    end_frame: int = 10,
    frame_step: int = 3,
    max_points: int = 30000,  # Subsample for browser performance
):
    """Create interactive Plotly visualizations."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("LOADING DATA FOR PLOTLY VISUALIZATION")
    print("=" * 70)

    # Load parser for transform
    parser = FreeTimeParser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8
    )

    # Load with grid-tracked initialization
    init_data = load_multiframe_colmap_grid_tracked(
        data_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        transform=parser.transform
    )

    positions = init_data['positions'].numpy()
    velocities = init_data['velocities'].numpy()
    times = init_data['times'].numpy().squeeze()
    colors = init_data['colors'].numpy()

    print(f"\nLoaded {len(positions)} points")

    # Subsample for browser performance
    if len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[idx]
        velocities = velocities[idx]
        times = times[idx]
        colors = colors[idx]
        print(f"Subsampled to {max_points} points for browser performance")

    vel_mag = np.linalg.norm(velocities, axis=1)

    # Convert colors to hex for plotly
    colors_hex = ['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]

    # =========================================================================
    # 1. Static 3D Point Cloud with Velocity Heatmap
    # =========================================================================
    print("\n[1] Creating 3D velocity heatmap...")

    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=vel_mag,
            colorscale='Hot',
            colorbar=dict(title='Velocity<br>Magnitude'),
            opacity=0.8
        ),
        text=[f'Vel: {v:.4f}<br>Time: {t:.2f}' for v, t in zip(vel_mag, times)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Point Cloud - Velocity Magnitude Heatmap',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )

    html_path = output_path / 'velocity_heatmap_3d.html'
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # =========================================================================
    # 2. Point Cloud with Original Colors
    # =========================================================================
    print("\n[2] Creating 3D point cloud with original colors...")

    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_hex,
            opacity=0.8
        ),
        text=[f'Vel: {v:.4f}<br>Time: {t:.2f}' for v, t in zip(vel_mag, times)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Point Cloud - Original Colors',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )

    html_path = output_path / 'pointcloud_colors_3d.html'
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # =========================================================================
    # 3. Animation: Point Cloud Moving Over Time
    # =========================================================================
    print("\n[3] Creating motion animation with time slider...")

    n_frames = 21  # Number of time steps
    time_values = np.linspace(0, 1, n_frames)

    # Create frames for animation
    frames = []
    for i, t in enumerate(time_values):
        # Apply motion: pos(t) = pos_init + vel * (t - mu_t)
        dt = t - times
        moved_positions = positions + velocities * dt[:, np.newaxis]

        frame = go.Frame(
            data=[go.Scatter3d(
                x=moved_positions[:, 0],
                y=moved_positions[:, 1],
                z=moved_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=vel_mag,
                    colorscale='Hot',
                    opacity=0.8
                ),
            )],
            name=f't={t:.2f}'
        )
        frames.append(frame)

    # Initial frame (t=0)
    dt_init = 0.0 - times
    moved_init = positions + velocities * dt_init[:, np.newaxis]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=moved_init[:, 0],
            y=moved_init[:, 1],
            z=moved_init[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=vel_mag,
                colorscale='Hot',
                colorbar=dict(title='Velocity'),
                opacity=0.8
            ),
        )],
        frames=frames
    )

    # Add slider and play button
    fig.update_layout(
        title='Point Cloud Motion Over Time (drag slider or press Play)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            xaxis=dict(range=[-6, 6]),
            yaxis=dict(range=[-6, 6]),
            zaxis=dict(range=[-6, 6]),
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                yanchor='top',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            mode='immediate'
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate'
                        )]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    font=dict(size=16),
                    prefix='Time: ',
                    visible=True,
                    xanchor='right'
                ),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[f't={t:.2f}'], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate'
                        )],
                        label=f'{t:.2f}',
                        method='animate'
                    )
                    for t in time_values
                ]
            )
        ],
        width=1200,
        height=900
    )

    html_path = output_path / 'motion_animation_3d.html'
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # =========================================================================
    # 4. 2D Views with Animation (Top-Down and Side)
    # =========================================================================
    print("\n[4] Creating 2D motion animation...")

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Top-Down View (X-Z)', 'Front View (X-Y)'],
        horizontal_spacing=0.1
    )

    # Initial positions at t=0
    dt_init = 0.0 - times
    moved_init = positions + velocities * dt_init[:, np.newaxis]

    # Add initial traces
    fig.add_trace(
        go.Scatter(
            x=moved_init[:, 0],
            y=moved_init[:, 2],
            mode='markers',
            marker=dict(size=2, color=vel_mag, colorscale='Hot', opacity=0.6),
            name='Top-Down'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=moved_init[:, 0],
            y=moved_init[:, 1],
            mode='markers',
            marker=dict(size=2, color=vel_mag, colorscale='Hot', opacity=0.6),
            name='Front'
        ),
        row=1, col=2
    )

    # Create frames
    frames = []
    for i, t in enumerate(time_values):
        dt = t - times
        moved = positions + velocities * dt[:, np.newaxis]

        frame = go.Frame(
            data=[
                go.Scatter(x=moved[:, 0], y=moved[:, 2], mode='markers',
                          marker=dict(size=2, color=vel_mag, colorscale='Hot', opacity=0.6)),
                go.Scatter(x=moved[:, 0], y=moved[:, 1], mode='markers',
                          marker=dict(size=2, color=vel_mag, colorscale='Hot', opacity=0.6)),
            ],
            name=f't={t:.2f}'
        )
        frames.append(frame)

    fig.frames = frames

    # Update layout
    fig.update_layout(
        title='2D Motion Animation',
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.1,
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix='Time: ', visible=True),
                pad=dict(t=50),
                steps=[
                    dict(args=[[f't={t:.2f}'], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                         label=f'{t:.2f}', method='animate')
                    for t in time_values
                ]
            )
        ],
        width=1400,
        height=700
    )

    # Set axis ranges
    fig.update_xaxes(range=[-5, 5], row=1, col=1)
    fig.update_yaxes(range=[-5, 5], row=1, col=1)
    fig.update_xaxes(range=[-5, 5], row=1, col=2)
    fig.update_yaxes(range=[-5, 5], row=1, col=2)

    html_path = output_path / 'motion_animation_2d.html'
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # =========================================================================
    # 5. Velocity Vector Field (sampled)
    # =========================================================================
    print("\n[5] Creating velocity vector field...")

    # Subsample for vector visualization
    vec_idx = np.random.choice(len(positions), min(3000, len(positions)), replace=False)

    # Create cone plot for velocity vectors
    fig = go.Figure(data=go.Cone(
        x=positions[vec_idx, 0],
        y=positions[vec_idx, 1],
        z=positions[vec_idx, 2],
        u=velocities[vec_idx, 0],
        v=velocities[vec_idx, 1],
        w=velocities[vec_idx, 2],
        colorscale='Hot',
        sizemode='absolute',
        sizeref=0.1,
        colorbar=dict(title='Velocity'),
    ))

    fig.update_layout(
        title='Velocity Vector Field (3D Cones)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )

    html_path = output_path / 'velocity_vectors_3d.html'
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # =========================================================================
    # 6. Points Colored by Time (mu_t)
    # =========================================================================
    print("\n[6] Creating point cloud colored by time...")

    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=times,
            colorscale='Viridis',
            colorbar=dict(title='Time (μt)'),
            opacity=0.8
        ),
        text=[f'Time: {t:.2f}<br>Vel: {v:.4f}' for t, v in zip(times, vel_mag)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D Point Cloud - Colored by Time (μt)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )

    html_path = output_path / 'pointcloud_time_3d.html'
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PLOTLY VISUALIZATIONS CREATED")
    print("=" * 70)
    print(f"Open these HTML files in your browser:")
    print(f"  1. {output_path / 'velocity_heatmap_3d.html'} - Velocity magnitude heatmap")
    print(f"  2. {output_path / 'pointcloud_colors_3d.html'} - Original colors")
    print(f"  3. {output_path / 'motion_animation_3d.html'} - 3D motion animation with slider")
    print(f"  4. {output_path / 'motion_animation_2d.html'} - 2D motion animation")
    print(f"  5. {output_path / 'velocity_vectors_3d.html'} - Velocity vector field")
    print(f"  6. {output_path / 'pointcloud_time_3d.html'} - Points colored by time")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="velocity_debug")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=10)
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=30000)
    args = parser.parse_args()

    visualize_velocity_plotly(
        args.data_dir,
        args.output_dir,
        args.start_frame,
        args.end_frame,
        args.frame_step,
        args.max_points,
    )
