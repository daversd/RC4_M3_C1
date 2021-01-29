using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VoxelState : MonoBehaviour
{
    public Voxel Voxel { get; private set; }

    public bool Marked { get; private set; }

    public float MarkValue; /*{ get; private set; }*/

    public void SetVoxel(Voxel voxel)
    {
        Voxel = voxel;
        Voxel.SetVoxelState(this);
    }

    public void Mark(float val = 1f)
    {
        Marked = true;
        MarkValue += val;
        MarkValue = Mathf.Clamp(MarkValue, 0f, 1f);
        Voxel.MarkVoxel(MarkValue);
        
    }

    public void Unmark()
    {
        MarkValue = 0f;
        Marked = false;
        Voxel.UnmarkVoxel();
    }
}
