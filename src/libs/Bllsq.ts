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
 * Typescript Math Toolkit.  Linear least squares with bagging and sub-bagging.
 *
 * @author Jim Armstrong (www.algorithmist.net)
 *
 * @version 1.0
 */
import { TSMT$LLSQ    } from "./llsq";
import { ILLSQResult  } from "./llsq";
import { TSMT$Bagging } from "./Bagging";
import { ISamples     } from "./Bagging";

export interface IBagggedLinearFit
{
  a: number;

  b: number;

  fits: Array<ILLSQResult>
}

export class TSMT$Bllsq
{
  constructor()
  {
    // empty
  }

 /**
  * Perform a linear regression (least squares fit) with bagged data sets
  *
  * @param {Array<number>} x Array of x-coordinates (must have at least three data points)
  *
  * @param {Array<number>} y Array of y-coordinates (must have at least three data points)
  *
  * @param {number} numSets Number of data sets or bags to use in the analysis
  *
  * @returns {IBaggedLinearFit}
  */
  public static bagFit(x: Array<number>, y: Array<number>, numSets: number): IBagggedLinearFit
  {
    const empty: IBagggedLinearFit = {
      a: 0,
      b: 0,
      fits: []
    };

    if (!x || !y) {
      return empty;
    }

    const n: number = x.length;
    if (n < 3) {
      return empty;
    }

    numSets = numSets == undefined || isNaN(numSets) || numSets < 1 ? n : Math.round(numSets);

    let a_ave: number = 0.0;
    let b_ave: number = 0.0;

    let i: number = 0;

    let fitArray: Array<ILLSQResult> = new Array<ILLSQResult>();
    let fit: ILLSQResult;

    const bag: Array<ISamples> = TSMT$Bagging.get2DSamplesWithReplacement( x, y, numSets );

    for (i = 0; i < numSets; ++i)
    {
      fit = TSMT$LLSQ.fit( bag[i].x, bag[i].y );

      a_ave += fit.a;
      b_ave += fit.b;

      fitArray.push(fit);
    }

    a_ave /= numSets;
    b_ave /= numSets;

    return {a: a_ave, b: b_ave, fits: fitArray};
  }

 /**
  * Perform a linear regression (least squares fit) with sub-bagged data sets
  *
  * @param {Array<number> x Array of x-coordinates (must have at least three data points)
  *
  * @param {Array<number>} y Array of y-coordinates (must have at least three data points)
  *
  * @param {number} m Number of original data points to use in a bag (must be less than or equal to total number of samples)
  *
  * @param {number} numSets Number of data sets or bags to use in the analysis
  *
  * @returns {IBaggedLinearFit}
  */
  public static subbagFit(x: Array<number>, y: Array<number>, m: number, numSets: number): IBagggedLinearFit
  {
    // TODO need to be more DRY
    const empty: IBagggedLinearFit = {
      a: 0,
      b: 0,
      fits: []
    };

    if (!x || !y) {
      return empty;
    }

    const n: number = x.length;
    if (n < 3) {
      return empty;
    }

    m       = m == undefined || isNaN(m) || m < 1 || m > n ? Math.floor(n/2) : Math.round(m);
    numSets = numSets == undefined || isNaN(numSets) || numSets < 1 ? n : Math.round(numSets);

    let a_ave: number = 0.0;
    let b_ave: number = 0.0;

    let fitArray: Array<ILLSQResult> = new Array<ILLSQResult>();
    let fit: ILLSQResult;
    let i: number;

    const bag: Array<ISamples> = TSMT$Bagging.get2DSamplesWithoutReplacement(x, y, m, numSets);

    for (i = 0; i < numSets; ++i )
    {
      fit   = TSMT$LLSQ.fit( bag[i].x, bag[i].y );
      a_ave += fit.a;
      b_ave += fit.b;

      fitArray.push(fit);
    }

    a_ave /= numSets;
    b_ave /= numSets;

    return {a: a_ave, b: b_ave, fits: fitArray};
  }
}
