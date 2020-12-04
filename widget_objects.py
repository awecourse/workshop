from ipywidgets import widgets, Layout

sliders = {
    'rf': widgets.FloatSlider(
        value=.2,
        min=-1,
        max=2,
        step=0.01,
        description='Reel-out factor [-]:',
        style={'description_width': 'initial'},
        continuous_update=False,
        # layout=Layout(width='400px', height='50px')
    ),
    'le': widgets.FloatSlider(
        value=10,
        min=0,
        max=500,
        step=10,
        description='Tether length [m]:',
        style={'description_width': 'initial'},
        continuous_update=False,
    ),
    'phi': widgets.FloatSlider(
        value=0,
        min=-90,
        max=90,
        step=1,
        description='Azimuth angle [deg]:',
        style={'description_width': 'initial'},
        continuous_update=False,
    ),
    'beta': widgets.FloatSlider(
        value=0,
        min=0,
        max=90,
        step=1,
        description='Elevation angle [deg]:',
        style={'description_width': 'initial'},
        continuous_update=False,
    ),
    'chi': widgets.FloatSlider(
        value=0,
        min=0,
        max=360,
        step=1,
        description='Course angle [deg]:',
        style={'description_width': 'initial'},
        continuous_update=False,
    ),
}