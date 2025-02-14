package caseStudyV1_1;

import java.util.Scanner;

public class Floor
{
    Slot s[][];
    Entry en[];
    Exit ex[];
    public boolean IsFloorEmpty;
    public int cost;
    private Scanner s1;
    private int costPerUnitCharge;
    Floor(int NoOfSlotsX,int NoOfSlotsY,int NoOfEntries,int NoOfExits, int cost,int NoOfCars,int NoOfBikes,int NoOfTrucks,int costPerUnitCharge)
    {
        this.costPerUnitCharge=costPerUnitCharge;
        this.s = new Slot[NoOfSlotsX][NoOfSlotsY];
        int k=1;
        for(int i=0;i<NoOfSlotsX;i++)
        {
            for(int j=0;j<NoOfSlotsY;j++)
            {
                if(k>0 && k<= NoOfBikes)
                {
                    s[i][j] = new Slot(k++,true,1);
                }
                else if(k>NoOfBikes && k<= NoOfBikes+NoOfCars)
                {
                    s[i][j] = new Slot(k++,true,2);
                }
                else
                {
                    s[i][j] = new Slot(k++,true,4);
                }
            }
        }
        this.en = new Entry[NoOfEntries];
        for(int i=0;i<NoOfEntries;i++)
        {
            en[i] = new Entry(i+1,true);
        }
        this.ex = new Exit[NoOfExits];
        for(int i=0;i<NoOfExits;i++)
        {
            ex[i] = new Exit(i+1,true);
        }
        this.IsFloorEmpty=true;
        this.cost=cost;
    }
    public void DisplaySlots()
    {
        for(int i=0;i<this.s.length;i++)
        {
            for(int j =0;j<this.s[0].length;j++)
            {
                if(s[i][j].IsSlotEmpty)
                {
                    System.out.print(" |Slot"+" "+s[i][j].getSlotId()+" "+s[i][j].RemainingSpace+"Space Empty| ");
                }
                else
                {
                    System.out.print(" |Slot"+" "+s[i][j].getSlotId()+" "+"Occupied| ");
                }
            }
            System.out.println();
        }
    }
    public void DisplayEntries()
    {
        for(int j =0;j<this.en.length;j++)
        {
            if(en[j].IsEmpty)
            {
                System.out.print(" |Entry"+(j+1)+"Empty| ");
            }
            else
            {
                System.out.print(" |Entry"+(j+1)+"Occupied| ");
            }
        }
        System.out.println();
    }
    public void DisplayExits()
    {
        for(int j =0;j<this.ex.length;j++)
        {
            if(ex[j].IsEmpty)
            {
                System.out.print(" |Exit"+(j+1)+"Empty| ");
            }
            else
            {
                System.out.print(" |Exit"+(j+1)+"Occupied| ");
            }
        }
        System.out.println();
    }
    protected int  AllocateParkingLot(Vehicle v)
    {
        if(this.IsFloorEmpty)
        {
            int space = v.getVehicleSpace();
            Scanner s12 = new Scanner(System.in);
            System.out.println("Enter your choice Of Slot");
            int getChoice = s12.nextInt();
            System.out.println("Your Choice is "+getChoice);
            for(int i=0;i<this.s.length;i++)
            {
                for(int j =0;j<this.s[0].length;j++)
                {
                    if(getChoice==s[i][j].getSlotId())
                    {
                        if(s[i][j].RemainingSpace < space)
                        {
                            System.out.println("Can't allocate due to insufficent Space");

                            return AllocateParkingLot(v);
                        }
                        else
                        {
                            v.AllocatedParkingLot=getChoice;

                            return getChoice;
                        }
                    }
                }
            }
        }

        return -1;
    }
    public boolean CheckFloorEmpty()
    {
        boolean ReturnValue = false;
        for(int i=0;i<this.s.length;i++)
        {
            for(int j =0;j<this.s[0].length;j++)
            {
                if(s[i][j].IsSlotEmpty)
                {
                    ReturnValue = true;
                    return ReturnValue;
                }
            }
        }
        return ReturnValue;
    }
public  int getCostperUnitCharge(){
    return costPerUnitCharge;
}
}
