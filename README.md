# Deep Learning With Angular 5, Typescript, and DeeplearnJS

If you are interested in deep learning, then congratulations because entry into this field now is a LOT easier than my first journey in the late 1990's.  The tools, techniques, and general knowledge base is far superior to the dark times .. before the Empire ...

All seriousness aside, the coolest tool to come along in a while is Google's _DeepLearnJS_ project.  If you have not already done so, [check it out here].

This project illustrates how to integrate _DeepLearnJS_ into an Angular/Typescript application.


Author:  Jim Armstrong - [The Algorithmist]

@algorithmist

theAlgorithmist [at] gmail [dot] com

Angular: 5.2

DeepLearnJS: 0.5.0

Angular CLI: 1.6.5

Dependency: EaselJS 0.83


## Introduction

One of the early examples presented on the _DeeplLearnJS_ site is that of fitting a quadratic polynomial to a small dataset.  I wanted to expand on this example, particularly by comparing a small-order polynomial fit (say linear to cubic) vs. the best-possible fit using [polynomial least squares].

I have the polynomial least squares code in the Typescript Math Toolkit, and we should be able to use the published example as a starting point for a comparision application in Angular.  Along the way, I hope to show how to integrate _DeeplearnJS_ into a Typescript application and work within the Angular framework.


## Datasets

Two mock datasets are provided in the _app/data/MockData.ts_ file.  The demo is setup to fit the first one by default.  In the _app/app.component.ts_ file, change the line,

```
this.__getPoints(this.SET1);
```   

to

```
this.__getPoints(this.SET2);
``` 

to switch datasets.

The original points are plotted on a 500x500 Canvas (using the _EaselJS_ library).  There is some logic in the application to map real coordinates in a y-up coordinate space to pixel coordinates in the y-down Canvas space.


## Least Squares

Since the name of the game is fitting a polynomial model, i.e. a0 + a1*x + a2*x^2 + ... ak*x^k to a fixed dataset, the least squares method provides a guaranteed optimal solution to a fit with minimum sum of squared residuals.  It is also reasonable for small-order polynomials.

In practice, I've never used the technique for any polynomial larger than fifth order.  The technique has proven to be moderately useful for interpolation and I simply don't trust it for extrapolation.  It does, however, provide a good baseline for comparing against a deep learning model.  Unlike the LS technique, DL has no idea that the presumed model is polynomial.

This demo provides methods for straight (textbook) linear least squares (via normal equations), as well as bagged and sub-bagged linear LS.  The Typescript Math Toolkit polynomial least squares class, _TSMT$Pllsq_ class is used to provided fits for second- through fourth-order polynomials.

The LS examples use the data transformed to Canvas coordinates, which I think is a bad idea in practice, but it's an easy habit to fall into.  Although we can 'get away' with it for this demo, it's an example of behavior I tend to call 'numerically risky.'  Notice that the polynomial coefficients are highly dominated by the constant term.


## DeeplearnJS

This is the crux of the demo.  Now, we could try to copy/pasta as much as possible from the quadratic-fit demo shown on the _DeeplearnJS_ site, but that's a mistake.  That demo works because the data is 'well-behaved' and already closely fits the presumed model.  If we tried the same technique with the SGD optimizer and the transformed Canvas data, it would not even work for a linear model.  The coefficients would speed off to infinity and beyond.

Using the original data works for a linear model, but not quadratic or higher.  Instead, I used the _RMSProp_ optimizer (which, quite frankly, I'm still learning about).  This works well for up to a quartic model, but oscillations become too high for a quintic or higher model, so I stopped at cubic for the demo.

Training and loss functions were separated out into a _DLModel_ class in the _app/dl-model_ folder.  I wanted to be very strict about typings, which I hope is helpful for those new to _DeeplearnJS_ and Typescript.

Model coefficients are initialized in the _ngOnInit()_ lifecycle method of the main app component, as this makes the initialization easy to locate.

```
this._dlVars = [
  dl.variable(dl.Scalar.new(Math.random())),
  dl.variable(dl.Scalar.new(Math.random())),
  dl.variable(dl.Scalar.new(Math.random())),
  dl.variable(dl.Scalar.new(Math.random()))
];
```

On my 2016 Macbook Pro, it took about 4-5 seconds to train the model (ymmv).  Since completion of training does not trigger Angular change detection, async pipes are used for all DL fit data updated in the view.  

The _Canvas_ selector directive causes the main app to get behind a CD cycle.  This can be resolved by forcing a check, or any of the various (and somewhat hacky) methods to wait a VM turn (timeout or RxJs delay).  The former is used for simplicity in this demo.

The fit curve from textbook least squares is plotted in blue; the deep learn curve is plotted in red.


## Build and Run

Nothing new here; `ng build` to build the project. `ng serve` build and run the server, then view the application on `localhost:4200`.

The initial display shows a plain, linear least squares fit in blue.  Depending on your hardware, the cubic deep learn fit will be displayed some time later, in red.  

![initial fit](images/dl-1.png?raw=true)

Use the dropdown to select different types of (textbook) least-squares fits.  Here is a cubic,

![initial fit](images/dl-2.png?raw=true)

Personally, I like the fit from _DeeplearnJS_ better as it impresses me as a more reliable model for extrapolation, especially as I created this dataset to illustrate the myopic nature of least squares :)

And, of course, deep learning is just plain cool, so experiment a bit with the demo and have fun learing _DeeplearnJS_ in tandem with Angular and Typescript.  But, most importantly, drink coffee!


## Further help

To get more help on the Angular CLI use `ng help` or go check out the [Angular CLI README](https://github.com/angular/angular-cli/blob/master/README.md).

[The Algorithmist]: <http://algorithmist.net>

[check it out here]: <https://deeplearnjs.org/index.html>

[polynomial least squares]: <http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html>
