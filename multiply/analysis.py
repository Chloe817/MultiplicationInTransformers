# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

pio.templates['plotly'].layout.xaxis.title.font.size = 20
pio.templates['plotly'].layout.yaxis.title.font.size = 20
pio.templates['plotly'].layout.title.font.size = 30
# pio.renderers.default = 'iframe'
print(f"Using renderer: {pio.renderers.default}")

# Import stuff
import torch
import numpy as np
import math
import transformer_lens.utils as utils


# from args import *

# Optional. Save graphs to files as PDF and HTML
save_graph_to_file = False
epochs_to_graph = 2000


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    fig = px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)
    fig.update_xaxes(title='Epoch', showgrid=False)
    fig.update_yaxes(title='Loss', showgrid=False)
    fig.show()


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


# Helper function to plot multiple lines
def lines(raw_lines_list, x=None, mode='lines', labels=None, xaxis='Epoch', yaxis='Loss', title = '', log_y=False, hover=None, all_epochs=True, **kwargs):
    global epochs_to_graph

    lines_list = raw_lines_list if all_epochs==False else [row[:epochs_to_graph] for row in raw_lines_list]

    log_suffix = '' if log_y==False else ' (Log)'
    epoch_suffix = '' if all_epochs==False else ' (' + str(epochs_to_graph) + ' Epochs)'
    full_title = title + log_suffix + epoch_suffix
    # full_title = ''
    # print(full_title)

    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x = np.arange(len(lines_list[0]))

    if save_graph_to_file :
      fig = go.Figure(layout={})
    else:
      fig = go.Figure(layout={'title':full_title})

    fig.update_xaxes(title=xaxis, showgrid=False)
    fig.update_yaxes(title=yaxis + log_suffix, showgrid=False)

    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = utils.to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))

    if log_y:
        fig.update_layout(yaxis_type="log")
    else:
        # Calculate the max y-value rounded up to the nearest integer
        y_max = 1
        for k in range(len(lines_list)):
            y_max = max(y_max, math.ceil(max(lines_list[k])) )
        y_max = 3.5 # manual override if necessary
        # Update layout to set the y-axis min to 0 and max to the calculated y_max
        fig.update_layout(yaxis=dict(range=[0, y_max]))

    # Update x-axis ticks
    x_ticks = x[0::300]  # Start from index 0 and pick every 100th element
    x_ticks = x_ticks[1:] # Exclude the first tick (0)
    fig.update_xaxes(tickmode='array', tickvals=x_ticks, ticktext=[str(tick) for tick in x_ticks])


    y_ticks = x[0::1]  # Start from index 0 and pick every 100th element
    y_ticks = y_ticks[1:] # Exclude the first tick (0)
    fig.update_yaxes(tickmode='array', tickvals=y_ticks, ticktext=[str(tick) for tick in y_ticks])

    fig.update_xaxes(tickfont=dict(size=20),)
    fig.update_yaxes(tickfont=dict(size=20),)
    fig.update_layout(legend=dict(
        font=dict(  # 图例字体
            # family="Courier",
            size=15,
            # color="red"  # 颜色：红色
        )))
    

    if save_graph_to_file:
      # fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),width=1200,height=300)
      # Update layout for legend positioning inside the graph
      fig.update_layout(
          margin=dict(l=10, r=10, t=10, b=10),
          width=1200,height=300,
          legend=dict(
              x=0.92,  # Adjust this value to move the legend left or right
              y=0.99,  # Adjust this value to move the legend up or down
              traceorder="normal",
              font=dict(
                  family="sans-serif",
                  size=12,
                  color="black"
              ),
              bgcolor="White",  # Adjust background color for visibility
              bordercolor="Black",
              borderwidth=2
          )
      )
    fig.show(bbox_inches="tight")
    # filename = full_title.replace(" ", "").replace("(", "").replace(")", "").replace("&", "").replace(",", "").replace("%", "")   +'.pdf' # or pdf, svg, png
    # pio.write_image(fig, filename)



# Graph per digit series using "normal" and "log" scale
def graph_perdigit(losslist, num_series, title_suffix, showlog, all_epochs=True):
    # lines([losslist[i] for i in range(num_series)],
    #         labels = [f'A{i}' for i in range(num_series)],
    #         title='Per digit '+title_suffix,
    #         all_epochs=all_epochs)

    # if showlog:
    #     lines([losslist[i] for i in range(num_series)],
    #         labels = [f'A{i}' for i in range(num_series)],
    #         title='Per digit '+title_suffix,
    #         all_epochs=all_epochs,
    #         log_y=True)
            
    if all_epochs==True :
        total_loss = 0
        final_loss_str = 'Final Loss: '
        mean_loss_str = 'Mean Loss : '
        for i in range(num_series):
            total_loss += losslist[i][-1]
            final_loss_str += 'A%s %.5f, '%(i, losslist[i][-1])
            mean_loss_str += 'A%s %.5f, '%(i, losslist[i].mean())
            # print('Final Loss for A%s is %.5f. Mean Loss is %.5f'%(i, losslist[i][-1], total_loss/num_series) )
        final_loss_str = final_loss_str[:-2]
        mean_loss_str = mean_loss_str + 'All %.5f'%(losslist.mean())
        print(final_loss_str)
        print(mean_loss_str)

