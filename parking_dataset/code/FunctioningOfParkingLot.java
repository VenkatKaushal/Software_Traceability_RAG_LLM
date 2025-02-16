package caseStudyV1_1;
import java.util.*;

public class FunctioningOfParkingLot {

    public static void PreliminaryInstructions()
    {
        System.out.println(" 1 To park Vehicle ");
        System.out.println(" 2 to remove Vehicle ");
        System.out.println(" -1 to end");
    }
    public static void Instructions()
    {
        System.out.println(" 1 To management view ");
        System.out.println(" 2 To customer view ");
        System.out.println(" -1 to end");
    }
    public static void ChooseVehicle()
    {
        System.out.println(" 1 To bike ");
        System.out.println(" 2 To car ");
        System.out.println(" 3 To truck ");
        System.out.println(" 4 To ElectricCar ");
    }


    public static void main(String[] args)
    {
        Scanner S2 = new Scanner(System.in);
        System.out.println("Enter Number Of Floors");
        int NoOfFloors = S2.nextInt();
        Management M = new Management(NoOfFloors);
        User U = new User();
        String Password = "PassWord";
        Instructions();
        int x9 = S2.nextInt();
        while(x9!= -1)
        {
            if(x9==1)
            {
                System.out.println("Enter Password");
                String GivenPassword = S2.next();
                if(Password.compareTo(GivenPassword)==0)
                {
                    System.out.println("Enter the floor and Slot of floor to Block ");
                    int Floor = S2.nextInt(),slotId = S2.nextInt();
                    M.ManageSlots(Floor,slotId,false);
                }
                else
                {
                    System.out.println("wrong password , Try again");
                }
            } else if (x9==2)
            {
                PreliminaryInstructions();
                int x = S2.nextInt();
                while(x!=-1)
                {
                    ChooseVehicle();
                    if(x==1)
                    {
                        System.out.println("Enter Choice of Vehicle and Floor Number Respectively");
                        int ChoiceOfVehicle = S2.nextInt(),FloorNumber=S2.nextInt();
                        U.Enter(M, FloorNumber-1, ChoiceOfVehicle);
                    }
                    else if(x==2)
                    {
                        int ChoiceOfVehicle = S2.nextInt();
                        U.Exit(M, ChoiceOfVehicle);
                    }
                    PreliminaryInstructions();
                    x =S2.nextInt();
                }
            }
            Instructions();
            x9 = S2.nextInt();
        }
        S2.close();
    }
}
