import * as dl from 'deeplearn';

/**
 * Predictor, training, and loss functions for a polynomial approximation to 2D data in the form a0 + a1*x + a2*x^2 + a3*x^3 + ...
 *
 * @author Jim Armstrong (www.algorithmist.net)
 *
 * @version 1.0
 */
export class DlModel
{
  /**
   * Predictor for a given input
   *
   * @param {number} input Value of independent variable
   *
   * @param {Array<Variable<dl.Rank.R0>>} params Model coefficients
   *
   * @returns {Scalar} Model value
   */
  public static predictor(input: number, params: Array< dl.Variable<dl.Rank.R0> >): dl.Scalar
  {
    return dl.tidy( () => {
      // independent variable
      const x: dl.Scalar = dl.Scalar.new(input);

      // evaluate polynomial using nested multiplication
      const n: number    = params.length;
      let val: dl.Scalar = params[n-1];
      let i: number;

      for (i = n-2; i >= 0; i--)
      {
        val = val.mul(x);
        val = val.add(params[i]);
      }

      return val;
    });
  }

  /**
   * Loss function for training
   *
   * @param {Scalar} prediction Predicted value
   *
   * @param {number} actual Actual value
   *
   * @returns {Scalar} Square of residual
   */
  public static loss(prediction: dl.Scalar, actual: number): dl.Scalar
  {
    // return square of residual
    const delta: dl.Scalar = dl.scalar(actual).sub(prediction);
    return delta.square();
  }

  public static async train(xtrain: Array<number>,
                            ytrain: Array<number>,
                            optimizer: dl.Optimizer,
                            params: Array< dl.Variable<dl.Rank.R0> >,
                            numIterations: number,
                            done: Function)
  {
    let iter: number;
    let i: number;

    const n: number = xtrain.length;

    for (iter = 0; iter < numIterations; iter++)
    {
      for ( i = 0; i < n; i++)
      {
        optimizer.minimize( () => {
          const pred: dl.Scalar = DlModel.predictor(xtrain[i], params);
          const loss: dl.Scalar = DlModel.loss(pred, ytrain[i]);

          return loss;
        });
      }

      await dl.nextFrame();  // does not block browser
    }

    // callback on complete
    done();
  }

  /**
   * Test the predictor on an array of input values and a trained set of model parameters
   *
   * @param {Array<number>} xValidate Validation set of x-coordinates
   *
   * @param {Array<Variable<dl.Rank.R0>>} params Model parameters
   *
   * @returns {Array<Scalar>} Predicted values for each of the input, independent variable values
   */
  public static test(xValidate: Array<number>, params: Array< dl.Variable<dl.Rank.R0> >): Array<dl.Scalar>
  {
    return dl.tidy( () => {
      const predicted: Array<dl.Scalar> = xValidate.map( (val: number) => { return DlModel.predictor(val, params) });

      return predicted;
    });
  }

  constructor()
  {
    // empty
  }
}
