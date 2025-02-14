package caseStudyV1_1;

abstract public class Vehicle
{
    protected int VehicleSpace;
    public int AllocatedParkingLot;
    public long InTime;
    public long OutTime;
    protected long BillPayed;
    public boolean IsFeePayed;
    protected String OwnerName;
    protected String PaymentMethod;
    protected void PayFee()
    {
        IsFeePayed = true;
    }
    public int getVehicleSpace()
    {
        return VehicleSpace;
    }
    public void ChargeVehicle(int z1,int c)
    {
        this.BillPayed -= (long)z1*c;
    }
}

