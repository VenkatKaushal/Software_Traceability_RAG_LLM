package caseStudyV1_1;

public class ElectricCar extends Vehicle {
    int charge;

    ElectricCar(String OwnerName){
        this.VehicleSpace = 2;
        this.OwnerName=OwnerName;
        this.IsFeePayed=false;
    }
    
}
