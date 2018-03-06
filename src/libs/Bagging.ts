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
 * Typescript Math Toolkit.  Some methods for assisting with rudimentary Bootstrap Aggregating (Bagging),
 * including sub-bagging
 *
 * @author Jim Armstrong (www.algorithmist.net)
 *
 * @version 1.0
 */
import {TSMT$Deviates} from "./Deviates";

export interface ISamples
{
  x: Array<number>;
  y: Array<number>;
}

export class TSMT$Bagging
{
  protected static __deviates: TSMT$Deviates  = new TSMT$Deviates();

  constructor()
  {
    // empty
  }

 /**
  * Return an array of samples of equivalent size to an input sample
  *
  * @param {Array<number>} data Array of numerical data (n observations)
  *
  * @param {number} numSets Total number of data sets to generate (B in many online references)
  *
  * @returns {Array<Array<number>>} Each array contains n uniformly sampled observations (with replacement) from
  * the original data set, which means that data can be repeated.
  */
  public static get1DSamplesWithReplacement(data: Array<number>, numSets: number): Array< Array<number> >
  {
    if (data === undefined || data == null || data.length == 0) {
      return [];
    }

    const n: number   = data.length;
    numSets = numSets == undefined || numSets == null || numSets < 1 ? n : Math.round(numSets);

    let i: number, j: number, index: number;
    let r: number = TSMT$Bagging.__deviates.uniform(1001, true);  // TODO allow this seed to be entered

    const output: Array< Array<number> > = new Array< Array<number> >();
    const min: number                    = -0.499;
    const max: number                    = n - 1 + 0.499;

    for (i = 0; i < numSets; ++i)
    {
      output[i] = new Array<number>();

      for (j = 0; j < n; ++j)
      {
        r     = TSMT$Bagging.__deviates.uniform(1001, false);
        index = Math.floor( Math.round(min + r*(max - min) ));
        index = Math.abs(index);  // sometimes computes -0

        output[i].push( data[index] );
      }
    }

    return output;
  }

 /**
  * Return an array of samples from an input data set of less than the original sample size
  *
  * @param {Array<number>} data Array of numerical data (n observations)
  *
  * @param {number} m New sample size (must be less than n - defaults to [n/2])
  *
  * @param {number} numSets Total number of data sets to generate (B in many online references)
  *
  * @returns {Array<Array<number>>} Each array contains uniformly sampled observations (WITHOUT replacement)
  * from the original data set (subbagging)
  *
  */
  public static get1DSamplesWithoutReplacement(data: Array<number>, m: number, numSets: number): Array< Array<number> >
  {
    if (data == undefined || data == null || data.length == 0) {
      return [];
    }

    const n: number = data.length;
    m               = m == undefined || isNaN(m) || m < 1 || m > n ? Math.floor(n/2) : Math.round(m);
    numSets         = numSets == undefined || isNaN(numSets) || numSets < 1 ? n : Math.round(numSets);

    let i: number, index: number;
    let r: number = TSMT$Bagging.__deviates.uniform(1001, true);  // TODO allow this seed to be entered

    const output: Array< Array<number> > = new Array< Array<number> >();
    const min: number                    = -0.499;
    const max: number                    = n - 1 + 0.499;

    // record indices that have already been sampled
    let sampledIndices: Array<boolean> = new Array<boolean>();

    for (i = 0; i < numSets; ++i)
    {
      output[i]             = new Array<number>();
      sampledIndices.length = 0;

      let tmp: Array<number> = output[i];

      while (tmp.length < m)
      {
        r     = TSMT$Bagging.__deviates.uniform(1001, false);
        index = Math.floor( Math.round(min + r*(max - min) ));
        index = Math.abs(index);  // sometimes computes -0

        if (sampledIndices[index] == undefined)
        {
          tmp.push( data[index] );
          sampledIndices[index] = true;
        }
      }
    }

    return output;
  }

 /**
  * Return an array of (2D) samples of equivalent size to an input sample
  *
  * @param {Array<number>} x Array of x-coordinates (length n)
  *
  * @param {Array<number>} y Array of y-coordinates (length n)
  *
  * @param {number} numSets Total number of data sets to generate (B in many online references)
  *
  * @returns {Array<ISamples>} Each element contains a uniformly sampled observation (with replacement) from the
  * original data set.  This does, however, mean that data can be repeated.  Returns empty array for bad input data.
  *
  */
  public static get2DSamplesWithReplacement(x: Array<number>, y: Array<number>, numSets: number): Array<ISamples>
  {
    if (!x || !y || x.length == 0) {
      return [];
    }

    const n: number = x.length;
    if (y.length != n) {
      return [];
    }

    numSets = numSets == undefined || isNaN(numSets) || numSets < 1 ? n : Math.round(numSets);

    let i: number, j: number, index: number;
    let r: number = TSMT$Bagging.__deviates.uniform(1001, true);

    const output: Array<ISamples> = new Array<ISamples>();
    const min    = -0.499;
    const max    = n - 1 + 0.499;

    for (i = 0; i < numSets; ++i)
    {
      output[i] = {
        x: new Array<number>(),
        y: new Array<number>()
      };

      for (j = 0; j < n; ++j)
      {
        r     = TSMT$Bagging.__deviates.uniform(1001, false);
        index = Math.floor( Math.round(min + r*(max - min) ));
        index = Math.abs(index);  // sometimes computes -0

        output[i].x.push( x[index] );
        output[i].y.push( y[index] );
      }
    }

    return output;
  }

 /**
  * Return an array of (2D) samples with replacement
  *
  * @param {Array<number>} x Array of x-coordinates (length n)
  *
  * @param {Array<number>} y Array of y-coordinates (length n)
  *
  * @param {number} m Number of samples, m < n
  *
  * @param {number} numSets Total number of data sets to generate (B in many online references)
  *
  * @returns {Array<ISamples>} Each element contains a uniformly sampled observation (with replacement) from the
  * original data set.  Returns empty array for bad input data.
  */
  public static get2DSamplesWithoutReplacement(x: Array<number>, y: Array<number>, m: number, numSets: number): Array<ISamples>
  {
    if (!x || !y || x.length == 0) {
      return [];
    }

    const n: number = x.length;
    if (y.length != n) {
      return [];
    }

    m       = m == undefined || m < 1 || m > n ? Math.floor(n/2) : m;
    numSets = numSets == undefined || isNaN(numSets) || numSets < 1 ? n : Math.round(numSets);

    let i: number, j: number, index: number;
    let r: number = TSMT$Bagging.__deviates.uniform(1001, true);

    let output: Array<ISamples> = new Array<ISamples>();
    const min: number           = -0.499;
    const max: number          = n - 1 + 0.499;

    // record indices that have already been sampled
    let sampledIndices: Array<boolean> = new Array<boolean>();

    for (i = 0; i < numSets; ++i)
    {
      output[i]      = {
        x: new Array<number>(),
        y: new Array<number>()
      };

      sampledIndices.length = 0;

      while( output[i].x.length < m )
      {
        r     = TSMT$Bagging.__deviates.uniform(1001, false);
        index = Math.floor( Math.round(min + r*(max - min) ));
        index = Math.abs(index);  // sometimes computes -0

        if (sampledIndices[index] == undefined)
        {
          output[i].x.push( x[index] );
          output[i].y.push( y[index] );

          sampledIndices[index] = true;
        }
      }

    }

    return output;
  }
}
