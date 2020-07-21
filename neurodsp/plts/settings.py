"""Default settings for plots."""

###################################################################################################
###################################################################################################

## Define collections of style arguments
# Plot style arguments are those that can be defined on an axis object
AXIS_STYLE_ARGS = ['title', 'xlabel', 'ylabel', 'xlim', 'ylim']
# Line style arguments are those that can be defined on a line object
LINE_STYLE_ARGS = ['alpha', 'lw', 'linewidth', 'ls', 'linestyle',
                   'marker', 'ms', 'markersize']
# Custom style arguments are those that are custom-handled by the plot style function
CUSTOM_STYLE_ARGS = ['title_fontsize', 'label_size', 'tick_labelsize',
                     'legend_size', 'legend_loc']
STYLERS = ['axis_styler', 'line_styler', 'custom_styler']
STYLE_ARGS = AXIS_STYLE_ARGS + LINE_STYLE_ARGS + CUSTOM_STYLE_ARGS + STYLERS

## Define default values for aesthetics
# These are all custom style arguments
TITLE_FONTSIZE = 20
LABEL_SIZE = 16
TICK_LABELSIZE = 16
LEGEND_SIZE = 12
LEGEND_LOC = 'best'
