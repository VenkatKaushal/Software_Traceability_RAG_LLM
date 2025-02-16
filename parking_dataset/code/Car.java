package caseStudyV1_1;

public class Car extends Vehicle
{
    Car(String OwnerName)
    {
        this.VehicleSpace = 2;
        this.OwnerName=OwnerName;
        this.IsFeePayed=false;
    }
}
