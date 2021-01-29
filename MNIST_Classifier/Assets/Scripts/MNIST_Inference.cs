using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class MNIST_Inference
{
    #region Fields and preferences

    // 12 The model asset used to create the inference model
    NNModel _modelAsset;

    // 13 The model used to run inferences
    Model _loadedModel;

    // 14 The worker that processes the inferences
    IWorker _worker;

    #endregion

    #region Constructor

    // 15 Create the class' constructor
    /// <summary>
    /// Constructor for the inference model object
    /// </summary>
    public MNIST_Inference()
    {
        // 16 Load the model asset from resources
        _modelAsset = Resources.Load<NNModel>("Models/MNIST_Model_00");
        
        // 17 Load the model
        _loadedModel = ModelLoader.Load(_modelAsset);

        // 18 Create the worker
        _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _loadedModel);
    }

    #endregion

    #region Public Methods

    // 19 Create the prediction method
    /// <summary>
    /// Run the inference model on an image
    /// </summary>
    /// <param name="image">The input image</param>
    /// <returns>The prediction's index</returns>
    public float Predict(Texture2D image)
    {
        // 20 Create a new tensor from the input image, single channel (greyscale)
        Tensor imageTensor = new Tensor(image, channels: 1);
        
        // 21 Run the worker on the tensor
        _worker.Execute(imageTensor);

        // 22 Get the result tensor, containing the prediction for all classes
        var outputTensor = _worker.PeekOutput();

        // 23 Get the index of the prediction with the highest probability
        float prediction = outputTensor.ArgMax()[0];

        // 24 Dispose used resources to free the GPU
        imageTensor.Dispose();
        outputTensor.Dispose();

        // 25 Return the inferred prediction
        return prediction;
    }

    #endregion
}
