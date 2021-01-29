using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;

public class EnvironmentManager : MonoBehaviour
{
    #region Fields
    
    VoxelGrid _voxelGrid;
    Vector3Int _gridSize = new Vector3Int(28, 1, 28);
    List<VoxelState> _voxelStates;

    Texture2D _preview;

    [SerializeField]
    Text _predictionText;

    bool _erasing = false;

    // 26 Add the Inference class
    MNIST_Inference _inferenceModel;

    #endregion

    #region Unity Methods

    void Start()
    {
        _preview = Resources.Load<Texture2D>("Materials/Preview/output");
        _voxelGrid = new VoxelGrid(_gridSize, transform.position, 1f);
        _voxelStates = new List<VoxelState>();
        foreach (var voxel in _voxelGrid.Voxels)
        {
            var go = voxel.SetParent(transform);
            _voxelStates.Add(go.GetComponent<VoxelState>());
        }
        UpdatePreview();

        // 27 Create the inference object
        _inferenceModel = new MNIST_Inference();
    }

    void Update()
    {
        // Use mouse to draw or erase on the grid
        if (Input.GetMouseButton(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);

            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                var objectHit = hit.transform;
                if (objectHit.CompareTag("Voxel"))
                {
                    var stateObject = objectHit.GetComponent<VoxelState>();
                    if (_erasing && stateObject.Marked)
                    {
                        stateObject.Unmark();
                        UpdatePreview();
                    }
                    else if (!_erasing)
                    {
                        stateObject.Mark();
                        var voxel = stateObject.Voxel;

                        Vector3Int ind = voxel.Index;
                        var indXp = ind + new Vector3Int(1, 0, 0);
                        var indXn = ind + new Vector3Int(-1, 0, 0);

                        var indZp = ind + new Vector3Int(0, 0, 1);
                        var indZn = ind + new Vector3Int(0, 0, -1);

                        var indD1 = ind + new Vector3Int(1, 0, 1);
                        var indD2 = ind + new Vector3Int(-1, 0, 1);

                        var indD3 = ind + new Vector3Int(1, 0, -1);
                        var indD4 = ind + new Vector3Int(-1, 0, -1);

                        if (Util.ValidateIndex(_gridSize, indXp)) _voxelGrid.Voxels[indXp.x, indXp.y, indXp.z].StateObj.Mark(0.05f);
                        if (Util.ValidateIndex(_gridSize, indXn)) _voxelGrid.Voxels[indXn.x, indXn.y, indXn.z].StateObj.Mark(0.05f);

                        if (Util.ValidateIndex(_gridSize, indZp)) _voxelGrid.Voxels[indZp.x, indZp.y, indZp.z].StateObj.Mark(0.05f);
                        if (Util.ValidateIndex(_gridSize, indZn)) _voxelGrid.Voxels[indZn.x, indZn.y, indZn.z].StateObj.Mark(0.05f);

                        if (Util.ValidateIndex(_gridSize, indD1)) _voxelGrid.Voxels[indD1.x, indD1.y, indD1.z].StateObj.Mark(0.05f);
                        if (Util.ValidateIndex(_gridSize, indD2)) _voxelGrid.Voxels[indD2.x, indD2.y, indD2.z].StateObj.Mark(0.05f);
                        if (Util.ValidateIndex(_gridSize, indD3)) _voxelGrid.Voxels[indD3.x, indD3.y, indD3.z].StateObj.Mark(0.05f);
                        if (Util.ValidateIndex(_gridSize, indD4)) _voxelGrid.Voxels[indD4.x, indD4.y, indD4.z].StateObj.Mark(0.05f);

                        // 11 Update the preview image
                        UpdatePreview();
                    }
                }
            }
        }

        // Use E key to cycle between draw and erase mode 
        if (Input.GetKeyDown(KeyCode.E))
        {
            _erasing = !_erasing;
            print($"Erasing is now {_erasing}");
        }

        // Use R key to reset the grid
        if (Input.GetKeyDown(KeyCode.R))
        {
            foreach (var voxel in _voxelStates)
            {
                voxel.Unmark();
            }
            UpdatePreview();
        }

        // 28 Check if the voxel grid is empty / unmarked
        if (!_voxelGrid.GetVoxels().ToArray().All(v => !v.IsMarked))
        {
            // 29 Run the ingerence model on the grid's image and store prediction
            var prediction = _inferenceModel.Predict(_voxelGrid.GridAsTexture());

            // 30 Write prediction result to UI
            _predictionText.text = $"Current prediction: {prediction}";
        }
        else
        {
            // 31 If the canvas is empty, don't run a prediction
            _predictionText.text = $"No prediction available";
        }
    }

    #endregion

    #region Private Methods

    // 08 Create the method to update the preview from the grid
    /// <summary>
    /// Úpdate the image preview of the grid
    /// </summary>
    void UpdatePreview()
    {
        // 09 Set the preview image's pixel to the grid's value
        _preview.SetPixels(_voxelGrid.GridAsTexture().GetPixels());

        // 10 Apply the changes to the image
        _preview.Apply();
    }

    #endregion
}