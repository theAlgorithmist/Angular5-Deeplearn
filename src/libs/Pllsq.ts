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
 * Typescript Math Toolkit.  Polynomial least squares (suitable for small-order polynomials).  Data should be reasonably
 * well-behaved (i.e. normal equations can be applied without numerical issues).
 *
 * @author Jim Armstrong (www.algorithmist.net)
 *
 * @version 1.0
 */
import { TSMT$Matrix } from "./Matrix";

export interface IPolyLLSQResult
{
  coef: Array<number>;

  rms: number;
}

export class TSMT$Pllsq
{
  protected _matrix: TSMT$Matrix;

  protected _c: Array<number>;
  protected _n: number;

  constructor()
  {
    this._matrix = new TSMT$Matrix();

    this._c = new Array<number>();
    this._n = 0;
  }

 /**
  * Perform a linear regression (least squares fit) using a polynomial of the specified order
  *
  * @param {Array<number>} x Array of x-coordinates (n data points)
  *
  * @param {Array<number>} y Array of y-coordinates (n data points)
  *
  * @param {number} m Order of polynomial (5 or lower is recommended) and n > m (defaults to 1 for invalid inputs)
  *
  * @return {IPolyLLSQResult} Fit model is c0 + c1*x + c2*x^2 + ... c(m-1)*x^m-1 - 'coef' property contains the array of
  * polynomial coefficients.  'r' property contains the r-squared or square of corr. coefficient value.  'rms' is the
  * square root of the average squared error between the LS estimator and the actual y-values (i.e. RMS error).
  */
  public fit(x: Array<number>, y: Array<number>, m: number): IPolyLLSQResult
  {
    const empty = {coef: [], r: 0, rms: 0};

    if (!x || !y) {
      return empty;
    }

    const n: number = x.length;
    m               = isNaN(m) || m < 1 ? 1 : Math.round(m) + 1;

    if (n <= m) {
      return empty;
    }

    // coefficient matrix
    const a: Array< Array<number> > = new Array< Array<number> >();
    let i: number, j: number;

    for (i = 0; i < m; ++i) {
      a[i] = new Array<number>();
    }

    // summations
    const asums: Array<number> = new Array<number>();
    const b: Array<number>     = [];
    const len: number          = 2*(m-1);

    asums[0] = n;
    for (i = 1; i <= len; ++i) {
      asums[i] = 0.0;
    }

    for (i = 0; i < m; ++i) {
      b[i] = 0.0;
    }

    let xj: number, yj: number, tmp: Array<number>;

    for (j = 0; j < n; ++j)
    {
      xj      = x[j];
      yj      = y[j];
      tmp     = new Array<number>();
      tmp[1]  = xj;
      b[0]   += yj;

      for (i = 2; i <= len; ++i) {
        tmp[i] = tmp[i - 1] * xj;
      }

      for (i = 1; i <= len; ++i)
      {
        asums[i] += tmp[i];

        if (i < m) {
          b[i] += tmp[i] * yj;
        }
      }
    }

    // assemble the matrix - technically symmetric (and probably positive-definite), but currently assembled as a
    // general dense matrix since order is expected to be very small
    let start: number = 0;
    let row: Array<number>;

    for (i = 0; i < m; ++i)
    {
      row = a[i];
      for (j = 0; j < m; ++j) {
        row[j] = asums[start + j];
      }

      start++;
    }

    // Solve Ac = b for the coefficient vector, c
    this._matrix.fromArray(a);
    this._c = this._matrix.solve(b);
    this._n = m;

    // rms error
    let s: number = 0.0;
    let t: number;

    for (i = 0; i < n; ++i)
    {
      t  = this.eval(x[i]) - y[i];
      s += t*t;
    }

    return { coef:this._c, rms:Math.sqrt(s/n) };
  }

  /**
   * Evaluate the LS polynomial at an input value
   *
   * @param {number} x x-coordinate at which the polynomial is to be evaluated
   *
   * @returns {number} LS polynomial evaluated at the specified x-coordinate, or zero if the fit() method is not called
   * before the evaluate method.
   */
  public eval(x: number): number
  {
    if (this._c.length == 0) {
      return 0.0;
    }

    const n: number = this._c.length;
    let val: number = this._c[n-1];
    let i: number;

    for (i = n-2; i >= 0; i--) {
      val = x * val + this._c[i];
    }

    return val;
  }
}
