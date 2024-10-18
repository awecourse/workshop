# tud-awe-workshop
Welcome to the workshop as part of the Airborne Wind Energy (AWE) course (AE4T40) of Delft University of Technology. Af the end of this workshop, participants should be able to:
* understand the limitations of Loyd's theory,
* understand how the kite operation affects the mean cycle power,
* understand the relation between the mean cycle power and the reeling speed,
* understand how the kite operation changes with wind speed.
 
For achieving these learning objectives, we calculate performance metrics of an AWE system using the assumptions underlying the [quasi-steady modelling framework](https://doi.org/10.1016/j.renene.2018.07.023 "Read more about the quasi-steady modelling framework in this paper.").
The workshop uses a Jupyter Notebook to walk you through the material. The notebook combines live Python code together with narrative text, which will guide you step-by-step through calculations and generate and visualize results along the way.
If you are new to Jupyter Notebooks, please watch a tutorial video on how to use them (e.g.: [this tutorial](https://youtu.be/HW29067qVWk?t=323 "Introduction video on Jupyter Notebook - watch from 5:23 to 14:25") - watch 5:23-14:25). Launch the Jupyter Notebook by clicking on the lower button:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/awecourse/workshop/HEAD?filepath=awe_workshop.ipynb)

### Providing feedback

Please take a minute to give feedback on the workshop. [Click here](https://docs.google.com/forms/d/e/1FAIpQLSen-pcbHG2a4kls6EONPkKxtgDshKIwKU7EY6aJ4BWSPtfXJA/viewform?usp=sf_link "Open feedback form") to open the feedback form.

### Optional: using the notebook offline

Launching the notebook directly from this repository using the button above is the preferred way. Alternatively you can run the notebook locally. However, you need to run it in Python 3.6 as some of the functionalities stopped working with newer versions. To run the notebook locally, clone this repository, or manually download it to your working folder (extract the .zip archive) if you are not
familiar with git. If you're new to Python, we recommend you to install
it using [Anaconda](https://docs.anaconda.com/anaconda/install/ "Installation instructions for Anaconda"). Anaconda comes with Jupyter Notebook and required
packages already installed. If you are not using Anaconda, please install [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html "Installation instructions for Jupyter Notebook") yourself and execute the lower command to
install the required packages:

```commandline
pip install --user scipy numpy matplotlib ipywidgets
```

In Windows you can launch Jupyter Notebook simply via the start menu. In Linux use the lower command in a terminal:

```commandline
jupyter notebook
```

If everything went well, Jupyter Notebook should now be opened in your browser. Navigate to the working folder and open the ipynb-file. If you have any problems, please send an email to r.schmehl@tudelft.nl.
