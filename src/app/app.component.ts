/**
 * Copyright 2018 Jim Armstrong (www.algorithmist.net)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Deep learning vs. linear least squares (with bagging) and polynomial least squares ... let the battle begin ...
 *
 * NOTE: Everything is placed in one component to make this analysis easier to deconstruct; more modularity would
 * be recommended in practice.
 *
 * @author Jim Armstrong (www.algorithmist.net)
 *
 * @version 1.0
 */

// platform imports
import {
  Component
  , OnInit
  , AfterViewInit
  , ViewChild
  , ChangeDetectorRef
  , ChangeDetectionStrategy
} from '@angular/core';

// all libraries used in the analysis
import { TSMT$LLSQ
       , ILLSQResult     } from "../libs/llsq";
import { IBagggedLinearFit
       , TSMT$Bllsq      } from "../libs/Bllsq";
import { TSMT$Pllsq
       , IPolyLLSQResult } from "../libs/Pllsq";

// canvas selector
import { CanvasSelectorDirective } from "./canvas-selector/canvas-selector.directive";

// CreateJS Suite
import * as createjs from 'createjs-module';

// RxJS
import { Observable      } from "rxjs/Observable";
import { BehaviorSubject } from "rxjs/BehaviorSubject";

// DeepLearn
import * as dl from 'deeplearn';

// Model to be used in for training/evaluation
import { DlModel } from "./dl-model/dlModel";

// and, we need some data
import { MockData } from "./data/MockData";


@Component({
  selector: 'app-root',

  templateUrl: './app.component.html',

  styleUrls: ['./app.component.css'],

  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppComponent implements OnInit, AfterViewInit
{
  // horizontal and vertical buffer space (in px)
  protected BUFFER: number = 5;

  // map fit type to a nice string
  public fitName: Array<Object>;

  // RMS error for a given fit
  public error$: Observable<number>;
  protected _errorSubject: BehaviorSubject<number>;

  // poly-fit coefs and deep-learning polynomial coefs; note that the model is of the form q0 + q1*x + q2*x^2 + ...
  public coef$: Observable< Array<number> >;
  protected _coefSubject: BehaviorSubject< Array<number> >;

  public dlCoefs: Array<number>;
  protected _dlVars: Array< dl.Variable<dl.Rank.R0> >;

  // learning status text
  public dlStatus$: Observable<string>;
  protected _statusSubject: BehaviorSubject<string>;

  // indicate which data set to plot and which fit to apply
  protected SET1: string         = 'DATASET_1';
  protected SET2: string         = 'DATASET_2';
  protected LLSQ: string         = 'LLSQ';
  protected LLSQ_BAG: string     = 'LLSQ_BAG';
  protected LLSQ_SUB_BAG: string = 'LLSQ_SUB_BAG';
  protected QUAD_LLSQ: string    = 'QUAD_LSQ';
  protected CUBIC_LLSQ: string   = 'CUBIC_LSQ';
  protected QUARTIC_LLSQ: string = 'QUARTIC_LSQ';

  // fit type
  protected _fitType: string;

  @ViewChild(CanvasSelectorDirective) _surface: CanvasSelectorDirective;

  // EaselJS
  protected _stage: createjs.Stage;
  protected _width: number;
  protected _height: number;
  protected _points: createjs.Shape;
  protected _fit: createjs.Shape;
  protected _deep: createjs.Shape;

  // data to plot/fit
  protected _x: Array<number>;
  protected _y: Array<number>;
  protected _xOrig: Array<number>;
  protected _yOrig: Array<number>;
  protected _xmin: number;
  protected _xmax: number;
  protected _ymin: number;
  protected _ymax: number;

  // quintic coefficients
  protected _a0: number;
  protected _a1: number;
  protected _a2: number;
  protected _a3: number;
  protected _a4: number;

  // polynomial LS fit
  protected _polyFit: TSMT$Pllsq;

  // training and validation sets for DL
  protected _trainX: Array<number>;
  protected _trainY: Array<number>;
  protected _validateX: Array<number>;
  protected _validateY: Array<number>;

  // learning rate and optimizer
  protected _learningRate: number;
  protected _optimizer: dl.Optimizer;

  constructor(protected _cd: ChangeDetectorRef)
  {
    this._a0   = 0;
    this._a1   = 0;
    this._a2   = 0;
    this._a3   = 0;
    this._a4   = 0;
    this._xmin = 0;
    this._xmax = 0;
    this._ymin = 0;
    this._ymax = 0;

    this.fitName = [
      {name: 'LLSQ', label: 'Linear Least Squares'},
      {name: 'LLSQ_BAG', label: 'Linear Least Squares Bagged'},
      {name: 'LLSQ_SUB_BAG', label: 'Linear Least Squares Sub-bagged'},
      {name: 'QUAD_LSQ', label: 'Quadratic Least Squares'},
      {name: 'CUBIC_LSQ', label: 'Cubic Least Squares'},
      {name: 'QUARTIC_LSQ', label: 'Quartic Least Squares'}
    ];

    this._fitType = this.LLSQ;

    this._trainX    = new Array<number>();
    this._trainY    = new Array<number>();
    this._validateX = new Array<number>();
    this._validateY = new Array<number>();

    this._polyFit = new TSMT$Pllsq();

    this._learningRate = 0.005;
    this._optimizer    = dl.train.rmsprop(this._learningRate);

    this._statusSubject = new BehaviorSubject<string>('DeepLearn training in progress ...');
    this.dlStatus$      = this._statusSubject.asObservable();

    this._errorSubject = new BehaviorSubject<number>(0);
    this.error$        = this._errorSubject.asObservable();

    this._coefSubject = new BehaviorSubject( new Array<number>() );
    this.coef$        = this._coefSubject.asObservable();
  }

  /**
   * Angular lifecycle method - on init
   *
   * @returns {nothing}
   */
  public ngOnInit(): void
  {
    // initialize the deep-learning model parameters to pseudo-random data; number of array elements controls the
    // degree of polynomial; no real reason to put this in the on-init handler other than it makes it very easy
    // to locate and change

    // cubic model - oscillations can get worse if you go higher
    this._dlVars = [
      dl.variable(dl.Scalar.new(Math.random())),
      dl.variable(dl.Scalar.new(Math.random())),
      dl.variable(dl.Scalar.new(Math.random())),
      dl.variable(dl.Scalar.new(Math.random()))
    ];
  }

  /**
   * Angular lifecycle method - after view init
   *
   * @returns {nothing}
   */
  public ngAfterViewInit(): void
  {
    // this lifecycle hook is used due to the canvas selector
    if (this._surface)
    {
      this._stage  = this._surface.createStage();
      this._width  = this._surface.width;
      this._height = this._surface.height;

      this.__easelJSSetup();

      // assign the training/validation sets
      this.__getPoints(this.SET1);
      this.__getTraining();

      // transform and then plot the points
      this._x = this.__transform(this._x, 'x');
      this._y = this.__transform(this._y, 'y');

      this.__plotPoints();

      // begin training the DL model
      DlModel.train(this._trainX, this._trainY, this._optimizer, this._dlVars, 100, () => {this.__onTrainingComplete()});

      // initialize the graph with a 'textbook' linear least squares fit
      this.fit(this.LLSQ);

      // and, at this point, we are behind a CD cycle - this is the cheap solution
      this._cd.detectChanges();
    }
  }

  // setup EaselJS environment
  protected __easelJSSetup()
  {
    this._points = new createjs.Shape();
    this._fit    = new createjs.Shape();
    this._deep   = new createjs.Shape();

    this._stage.addChild(this._points);
    this._stage.addChild(this._fit);
    this._stage.addChild(this._deep);
  }

  /**
   * Perform a least-squares fit of the specified type
   *
   * @param {string} type Fit type
   *
   * @returns {nothing}
   */
  public fit(type: string): void
  {
    // note that the fit is performed on the plotting-transformed data; we really should NOT do this, but can get away
    // with it in this demo; should definitely not do this for DL
    let g: createjs.Graphics = this._fit.graphics;
    g.clear();

    let x1: number, y1: number, x2: number, y2: number;

    switch (type)
    {
      case this. LLSQ:
        let fit: ILLSQResult = TSMT$LLSQ.fit(this._x, this._y);

        x1 = this._x[0];
        y1 = fit.a*x1 + fit.b;

        x2 = this._x[this._x.length-1];
        y2 = fit.a*x2 + fit.b;

        this.__drawFitLine(g, x1, y1, x2, y2, '#0000ff');

        this.__getError(fit);

        this._coefSubject.next( [fit.b, fit.a] );
      break;

      case this.LLSQ_BAG:
        let bagFit: IBagggedLinearFit = TSMT$Bllsq.bagFit(this._x, this._y, 6);

        x1 = this._x[0];
        y1 = bagFit.a*x1 + bagFit.b;

        x2 = this._x[this._x.length-1];
        y2 = bagFit.a*x2 + bagFit.b;

        this.__drawFitLine(g, x1, y1, x2, y2, '#0000ff');

        this.__getError(bagFit);

        this._coefSubject.next( [bagFit.b, bagFit.a] );
      break;

      case this.LLSQ_SUB_BAG:
        let subbagFit: IBagggedLinearFit = TSMT$Bllsq.subbagFit(this._x, this._y, 20, 6);

        x1 = this._x[0];
        y1 = subbagFit.a*x1 + subbagFit.b;

        x2 = this._x[this._x.length-1];
        y2 = subbagFit.a*x2 + subbagFit.b;

        this.__drawFitLine(g, x1, y1, x2, y2, '#0000ff');

        this.__getError(subbagFit);

        this._coefSubject.next( [subbagFit.b, subbagFit.a] );
      break;

      case this.QUAD_LLSQ:
        let quadFit: IPolyLLSQResult = this._polyFit.fit(this._x, this._y, 2);

        this.__drawFitCurve(g, this._x[0], this._x[this._x.length-1], '#0000ff');

        this.__getError(quadFit);

        this._coefSubject.next( quadFit.coef.slice() );
      break;

      case this.CUBIC_LLSQ:
        let cubicFit: IPolyLLSQResult = this._polyFit.fit(this._x, this._y, 3);

        this.__drawFitCurve(g, this._x[0], this._x[this._x.length-1], '#0000ff');

        this.__getError(cubicFit);

        this._coefSubject.next( cubicFit.coef.slice() );
      break;

      case this.QUARTIC_LLSQ:
        let quarticFit: IPolyLLSQResult = this._polyFit.fit(this._x, this._y, 4);

        this.__drawFitCurve(g, this._x[0], this._x[this._x.length-1], '#0000ff');

        this.__getError(quarticFit);

        this._coefSubject.next( quarticFit.coef.slice() );
      break;
    }

    this._stage.update();
  }

  // copy the original data and break out into training/validation sets; the latter is open for experimentation
  protected __getTraining(): void
  {
    this._xOrig = this._x.slice();
    this._yOrig = this._y.slice();

    // uncomment to use every other point, for example
    // const n: number = this._xOrig.length;
    // let i: number;
    //
    // for (i = 0; i < n; i+=2)
    // {
    //   this._trainX.push(this._xOrig[i]);
    //   this._trainY.push(this._yOrig[i]);
    //
    //   this._validateX.push(this._xOrig[i+1]);
    //   this._validateY.push(this._yOrig[i+1]);
    // }
    //
    // // any leftover gets placed into the training set
    // if (i+1 < n-1)
    // {
    //   this._trainX.push(this._xOrig[n-1]);
    //   this._trainY.push(this._yOrig[n-1]);
    // }

    this._trainX = this._x.slice();
    this._trainY = this._y.slice();
  }

  // compute the RMS error for a given fit
  protected __getError(fit: Object): void
  {
    let err: number = 0;
    const n: number = this._x.length;
    let i: number, y: number, d: number;

    if (this.__isLLSQ(fit) || this.__isBLLSQ(fit))
    {
      for (i = 0; i < n; ++i)
      {
        y = fit.a*this._x[i] + fit.b;
        d = y - this._y[i];

        err += d*d;
      }

      this._errorSubject.next( Math.sqrt(err/n) );
    }
    else if (this.__isPLLSQ(fit))
    {
      this._errorSubject.next( fit.rms );
    }
    else
    {
      // who put an extra fit in here?  Homey don't play that!
      console.log( "Invalid fit: ", fit );
    }
  }

  // type guard for ILLSQResult
  protected __isLLSQ(fit: Object): fit is ILLSQResult
  {
    return fit.hasOwnProperty('chi2');
  }

  // type guard for IBaggedLinearFit
  protected __isBLLSQ(fit: Object): fit is IBagggedLinearFit
  {
    return fit.hasOwnProperty('fits');
  }

  // type guard for IPolyLLSQResult
  protected __isPLLSQ(fit: Object): fit is IPolyLLSQResult
  {
    return fit.hasOwnProperty('coef');
  }

  // draw a fit line into the supplied graphic context
  protected __drawFitLine(g: createjs.Graphics, x1: number, y1: number, x2: number, y2: number, color: string): void
  {
    g.setStrokeStyle(2);
    g.beginStroke(color);
    g.moveTo(x1, y1);
    g.lineTo(x2, y2);
    g.endStroke();
  }

  // draw a fit curve into the supplied graphic context
  protected __drawFitCurve(g: createjs.Graphics, xMin: number, xMax: number, color: string): void
  {
    g.setStrokeStyle(2);
    g.beginStroke(color);

    let y: number = this._polyFit.eval(xMin);
    g.moveTo(xMin, y);

    let delta: number = 3;
    let x: number     = xMin + delta;

    while (x < xMax)
    {
      y = this._polyFit.eval(x);
      g.lineTo(x, y);

      x += delta;
    }

    y = this._polyFit.eval(xMax);
    g.lineTo(xMax, y);
    g.endStroke();
  }

  // return the mock data to use based on the specified dataset
  protected __getPoints(set: string): void
  {
    if (set == this.SET1)
    {
      this._x = MockData.DATASET_1.x.slice();
      this._y = MockData.DATASET_1.y.slice();
    }
    else
    {
      this._x = MockData.DATASET_2.x.slice();
      this._y = MockData.DATASET_2.y.slice();
    }
  }

  // plot the data points inside the current Canvas boundary (with buffer space)
  protected __plotPoints(): void
  {
    const g: createjs.Graphics = this._points.graphics;
    g.clear();

    let i: number;
    for (i = 0; i < this._x.length; ++i)
    {
      g.beginFill('#00ff00');
      g.drawCircle(this._x[i], this._y[i], 3);
      g.endFill();
    }
  }

  // transform the data from raw coords in a y-up system to y-down Canvas coordinates within the display width and
  // height of the current drawing surface
  protected __transform(arr: Array<number>, coord: string='x'): Array<number>
  {
    // get the min and max elements
    const n: number = arr.length;
    let min: number = arr[0];
    let max: number = arr[0];
    let i: number;

    for (i = 1; i < n; ++i)
    {
      min = Math.min(min, arr[i]);
      max = Math.max(max, arr[i]);
    }

    const range: number = max - min;
    let output: Array<number>;
    let m: number;

    // the linear equation is (y - y1) = m(x - x1) or y = m(x - x1) + y1
    if (coord == 'x')
    {
      this._xmin = min;
      this._xmax = max;

      // map [min-max] to [buffer,width-buffer]
      m = (this._width - 2*this.BUFFER) / (max - min);

      output = arr.map( (x: number) => {return Math.round( m*(x - min) + this.BUFFER )} );
    }
    else
    {
      this._ymin = min;
      this._ymax = max;

      // y-coordinates are a little different because of switching from y-up to y-down.  The mapping is
      // [min, max] to [height-buffer, buffer]
      m = (2*this.BUFFER - this._height) / (max - min);

      output = arr.map( (y: number) => {return Math.round( m*(y - min) + this._height - this.BUFFER )} );
    }

    return output;
  }

  // execute whenever DL training is complete
  protected __onTrainingComplete(): void
  {
    this.dlCoefs = this._dlVars.map( (s: dl.Scalar) => {return s.dataSync()[0]} );

    const g: createjs.Graphics = this._deep.graphics;

    g.clear();
    g.setStrokeStyle(2);
    g.beginStroke('#ff0000');

    // plot the curve - we have to transform the
    const n: number = this.dlCoefs.length;
    let xd: number, yd: number;

    const mx: number = (this._width - 2*this.BUFFER)  / (this._xmax - this._xmin);
    const my: number = (2*this.BUFFER - this._height) / (this._ymax - this._ymin);

    if (n == 2)
    {
      let x1: number = this._xmin;
      let y1: number = this.dlCoefs[0] + this.dlCoefs[1] * x1;

      let x2: number = this._xmax;
      let y2: number = this.dlCoefs[0] + this.dlCoefs[1] * x2;

      // transform to Canvas coordinates for display
      xd = Math.round(mx * (x1 - this._xmin) + this.BUFFER);
      yd = Math.round(my * (y1 - this._ymin) + this._height - this.BUFFER);

      g.moveTo(xd, yd);

      xd = Math.round(mx * (x2 - this._xmin) + this.BUFFER);
      yd = Math.round(my * (y2 - this._ymin) + this._height - this.BUFFER);

      g.lineTo(xd, yd);
    }
    else
    {
      // draw curve
      let delta: number = 0.1;
      let x: number, xc: number, yc;
      let y: number;
      let i: number;

      // transform to Canvas coordinates for display

      x = this._xmin;
      y = this.dlCoefs[n-1];

      for (i = n-2; i >= 0; i--) {
        y = x * y + this.dlCoefs[i];
      }

      xc = Math.round(mx * (x - this._xmin) + this.BUFFER);
      yc = Math.round(my * (y - this._ymin) + this._height - this.BUFFER);

      g.moveTo(xc, yc);

      x = this._xmin + delta;
      while (x < this._xmax)
      {
        // evaluate the polynomial with nested multiplication
        y = this.dlCoefs[n-1];

        for (i = n-2; i >= 0; i--) {
          y = x * y + this.dlCoefs[i];
        }

        xc = Math.round(mx * (x - this._xmin) + this.BUFFER);
        yc = Math.round(my * (y - this._ymin) + this._height - this.BUFFER);

        g.lineTo(xc, yc);

        x += delta;
      }

      x = this._xmax;
      y = this.dlCoefs[n-1];

      for (i = n-2; i >= 0; i--) {
        y = x * y + this.dlCoefs[i];
      }

      xc = Math.round(mx * (x - this._xmin) + this.BUFFER);
      yc = Math.round(my * (y - this._ymin) + this._height - this.BUFFER);

      g.lineTo(xc, yc);
    }

    g.endStroke();

    this._stage.update();

    this._statusSubject.next('DL training complete.');
  }
}
