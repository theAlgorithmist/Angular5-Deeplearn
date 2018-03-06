/**
 * Copyright 2016 Jim Armstrong (www.algorithmist.net)
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
 * Typescript Math Toolkit: Linear least squares analysis of x-y data
 *
 * @author Jim Armstrong (www.algorithmist.net)
 *
 * @version 1.0
 *
 */
export interface ILLSQResult
{
  a: number;     // fit model is ax + b

  b: number;     // fit model is ax + b

  siga: number;  // measure of uncertainty in the a-parameter  'r' is the square (R^2) of the

  sigb: number;  // measure of uncertainty in the b-parameter.

  chi2: number;  // chi-squared parameter for the fit

  r: number;     // square (R^2) of the correlation coefficient.
}

export class TSMT$LLSQ
{
  constructor()
  {
    // empty
  }

 /**
  * Perform a linear regression (least squares fit) without data on variance of individual sample y-coordinates
  *
  * @param {Array<number>} x Array of x-coordinates (must have at least three data points)
  *
  * @param {Array<number>} y Array of y-coordinates (must have at least three data points)
  *
  * @return {ILLSQResult} There should be at least three points in the data set.  Invalid inputs result in a
  * fit to a singleton point at the origin.
  *
  * Reference: NRC or Wikipedia (https://en.wikipedia.org/wiki/Simple_linear_regression)
  */
  public static fit(_x: Array<number>, _y: Array<number>): ILLSQResult
  {
    const n: number = _x.length;

    if (n < 3 || _y.length != n) {
      return {a: 0, b: 0, siga: 0, sigb: 0, chi2: 0, r: 0};
    }

    let a: number   = 0.0;
    let b: number   = 0.0;
    let s: number   = 0.0;
    let sx: number  = 0.0;
    let sy: number  = 0.0;
    let st2: number = 0.0;

    let i: number, t: number, w: number;

    for (i = 0; i < n; ++i)
    {
      sx += _x[i];
      sy += _y[i];
    }

    const ss: number    = n;
    const sxoss: number = sx / ss;
    const ybar: number  = sy / ss;

    for (i = 0; i < n; ++i)
    {
      t    = _x[i] - sxoss;
      st2 += t * t;
      b   += t * _y[i];
    }

    b /= st2;
    a  = (sy - sx * b) / ss;

    let sigdat: number = 1.0;

    let siga: number = Math.sqrt((1.0 + sx * sx / (ss * st2)) / ss);
    let sigb: number = Math.sqrt(1.0 / st2);

    let chi2: number = 0.0;
    for (i = 0; i < n; ++i)
    {
      w     = _y[i] - ybar;
      t     = _y[i] - a - b * _x[i];
      chi2 += t * t;
      s    += w * w;
    }

    if (n > 2) {
      sigdat = Math.sqrt(chi2 / (n - 2));
    }

    siga *= sigdat;
    sigb *= sigdat;

    const cov: number = -sx / st2;        // unused, but reserved for future use
    const r: number   = 1.0 - chi2 / s;

    return {a: b, b: a, siga: siga, sigb: sigb, chi2: chi2, r: r};
  }
}
