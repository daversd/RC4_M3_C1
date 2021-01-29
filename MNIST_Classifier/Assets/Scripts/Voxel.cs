using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class Voxel : IEquatable<Voxel>
{
    #region Public fields

    public Vector3Int Index;
    public List<Face> Faces = new List<Face>(6);
    public Vector3 Center => (Index + _voxelGrid.Origin) * _size;
    public bool IsActive;
    public bool IsMarked { get; private set; }
    public VoxelState StateObj { get; private set; }

    #endregion

    #region Protected fields

    protected GameObject _voxelGO;
    protected VoxelGrid _voxelGrid;
    protected float _size;

    #endregion

    #region Contructors

    /// <summary>
    /// Creates a regular voxel on a voxel grid
    /// </summary>
    /// <param name="index">The index of the Voxel</param>
    /// <param name="voxelgrid">The <see cref="VoxelGrid"/> this <see cref="Voxel"/> is attached to</param>
    /// <param name="voxelGameObject">The <see cref="GameObject"/> used on the Voxel</param>
    public Voxel(Vector3Int index, VoxelGrid voxelGrid, float sizeFactor = 1f)
    {
        Index = index;
        _voxelGrid = voxelGrid;
        _size = _voxelGrid.VoxelSize;
        _voxelGO = GameObject.CreatePrimitive(PrimitiveType.Cube);
        _voxelGO.transform.position = (_voxelGrid.Origin + Index) * _size;
        _voxelGO.transform.localScale *= _voxelGrid.VoxelSize * sizeFactor;
        _voxelGO.name = $"Voxel_{Index.x}_{Index.y}_{Index.z}";
        _voxelGO.tag = "Voxel";
        _voxelGO.GetComponent<MeshRenderer>().material = Resources.Load<Material>("Materials/Basic");
        var vState = _voxelGO.AddComponent<VoxelState>();
        vState.SetVoxel(this);
        vState.Unmark();
    }

    /// <summary>
    /// Generic constructor, alllows the use of inheritance
    /// </summary>
    public Voxel() { }

    #endregion

    #region Public methods

    public void MarkVoxel(float val = 1f)
    {
        IsMarked = true;
        if (val == 1f)
        {
            _voxelGO.GetComponent<MeshRenderer>().material = Resources.Load<Material>("Materials/Marked");
        }
        else if (val >= 0.8f)
        {
            _voxelGO.GetComponent<MeshRenderer>().material = Resources.Load<Material>("Materials/Grad4");
        }
        else if (val >= 0.6f)
        {
            _voxelGO.GetComponent<MeshRenderer>().material = Resources.Load<Material>("Materials/Grad3");
        }
        else if (val >= 0.4f)
        {
            _voxelGO.GetComponent<MeshRenderer>().material = Resources.Load<Material>("Materials/Grad2");
        }
        else if (val >= 0.2f)
        {
            _voxelGO.GetComponent<MeshRenderer>().material = Resources.Load<Material>("Materials/Grad1");
        }
    }

    public void UnmarkVoxel()
    {
        _voxelGO.GetComponent<MeshRenderer>().material = Resources.Load<Material>("Materials/Basic");
        IsMarked = false;
    }

    public GameObject SetParent(Transform parent)
    {
        _voxelGO.transform.parent = parent;
        return _voxelGO;
    }

    public void SetVoxelState(VoxelState stateObj)
    {
        StateObj = stateObj;
    }

    /// <summary>
    /// Get the neighbouring voxels at each face, if it exists
    /// </summary>
    /// <returns>All neighbour voxels</returns>
    public IEnumerable<Voxel> GetFaceNeighbours()
    {
        int x = Index.x;
        int y = Index.y;
        int z = Index.z;
        var s = _voxelGrid.GridSize;

        if (x != 0) yield return _voxelGrid.Voxels[x - 1, y, z];
        if (x != s.x - 1) yield return _voxelGrid.Voxels[x + 1, y, z];

        if (y != 0) yield return _voxelGrid.Voxels[x, y - 1, z];
        if (y != s.y - 1) yield return _voxelGrid.Voxels[x, y + 1, z];

        if (z != 0) yield return _voxelGrid.Voxels[x, y, z - 1];
        if (z != s.z - 1) yield return _voxelGrid.Voxels[x, y, z + 1];
    }

    /// <summary>
    /// Activates the visibility of this voxel
    /// </summary>
    public void ActivateVoxel(bool state)
    {
        IsActive = state;
        _voxelGO.GetComponent<MeshRenderer>().enabled = state;
        _voxelGO.GetComponent<BoxCollider>().enabled = state;
    }

    #endregion

    #region Equality checks
    /// <summary>
    /// Checks if two Voxels are equal based on their Index
    /// </summary>
    /// <param name="other">The <see cref="Voxel"/> to compare with</param>
    /// <returns>True if the Voxels are equal</returns>
    public bool Equals(Voxel other)
    {
        return (other != null) && (Index == other.Index);
    }

    /// <summary>
    /// Get the HashCode of this <see cref="Voxel"/> based on its Index
    /// </summary>
    /// <returns>The HashCode as an Int</returns>
    public override int GetHashCode()
    {
        return Index.GetHashCode();
    }

    #endregion
}