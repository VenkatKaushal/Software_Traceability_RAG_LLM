package caseStudyV1_1;
import java.util.*;

public class Management 
{
    Floor F[];
    LinkedList<Vehicle> ListVehicles;
    public Management(int NoOfFloors)
    {
        ListVehicles = new LinkedList<>();
        this.F= new Floor[NoOfFloors];
        for(int i=0;i<NoOfFloors;i++)
        {
            Scanner S1 =new Scanner(System.in);
            System.out.println("enter No of Entries,Exits,SlotsX,SlotY,Cost,NoOfCars,Trucks,Bikes,costPerUnitCharge Respectively");
            int Entries =S1.nextInt(),Exits=S1.nextInt(),SlotsX=S1.nextInt(),SlotsY = S1.nextInt(),cost =S1.nextInt(),NoOfCars=S1.nextInt(),NoOfTrucks=S1.nextInt(),NoOfBikes=S1.nextInt(),costPerCharge=S1.nextInt();
            F[i]= new Floor(SlotsX,SlotsY,Entries,Exits,cost,NoOfCars,NoOfBikes,NoOfTrucks,costPerCharge);
 //           S1.close();
        }
    }
   protected void ManageSlots(int FloorNumber,int SlotId,boolean SlotStatus)
    {
        int xcordinate = --SlotId/(F[FloorNumber-1].s[0].length);
        int ycordinate =SlotId%(F[FloorNumber-1].s[0].length);
        F[FloorNumber-1].s[xcordinate][ycordinate].SlotStatusSetter(SlotStatus);
    }
}
