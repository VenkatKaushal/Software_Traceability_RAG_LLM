package caseStudyV1_1;

public class Slot
{
   protected boolean IsSlotEmpty;
    private int SlotIdX;
    public int SlotSpace;
    public int RemainingSpace;
    public Slot(int SlotIdX, boolean IsSlotEmpty, int SlotSpace)
    {
        this.IsSlotEmpty = IsSlotEmpty;
        this.SlotIdX = SlotIdX;
        this.SlotSpace = SlotSpace;
        this.RemainingSpace=SlotSpace;
    }
    public int getSlotId()
    {
        return SlotIdX;
    }
    protected void MakeSlotFree(int VehicleSpace)
    {
        this.RemainingSpace+=VehicleSpace;
        this.IsSlotEmpty=true;
    }
    public void SlotStatusSetter(boolean Status)
    {
        this.IsSlotEmpty = Status;
    }
}
