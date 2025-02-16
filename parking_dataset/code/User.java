package caseStudyV1_1;

import java.util.Scanner;

public class User
{
    private String VehicleName;
    private String PlateNumber;

    public void VehicleNameSetter(String VehicleName)
    {
        this.VehicleName = VehicleName;
    }

    public String VehicleNameGetter()
    {
        return this.VehicleName;
    }
    public void PlateNumberSetter(String PlateNumber)
    {
        this.PlateNumber = PlateNumber;
    }

    public String PlateNumberGetter()
    {
        return this.PlateNumber;
    }
    
    protected void Enter(Management M, int FloorNumber,int ChoiceOfVehicle)
    {
        Scanner S = new Scanner(System.in);
        if(M.F[FloorNumber].CheckFloorEmpty())
                    { 
                        System.out.println("Enter your Full Name ");
                        String Name = S.next();
                        Vehicle B[] = new Vehicle[4];
                        B[0]= new Bike(Name);
                        B[1]= new Car(Name);
                        B[2]= new Truck(Name);
                        B[3]= new ElectricCar(Name);
                        M.F[FloorNumber].DisplayEntries();
                        int x3 = S.nextInt()-1;      //taking entry number
                        M.F[FloorNumber].en[x3].IsEmpty= false;
                        M.F[FloorNumber].en[x3].PaymentOptions();
                        int x4 = S.nextInt();// asking for pre payment
                        if(x4!= -1)
                        {
                            System.out.println(" Enter Advance You wish To pay ");
                            int Advance = S.nextInt();
                            M.F[FloorNumber].en[x3].PrePayment(Advance, B[ChoiceOfVehicle-1]);
                        }
                        M.F[FloorNumber].en[x3].AssignInTime(B[ChoiceOfVehicle-1]);// assign in time
                        M.F[FloorNumber].DisplaySlots();// displaying slots
                        if(ChoiceOfVehicle==4)// if electric vehicle ask for charge
                        {
                            System.out.println("Enter 1 to Charge vehicle or anthing else if not neede");
                            int r=S.nextInt();
                            if(r==1)
                            {
                               System.out.println("Enter units of charge required: ");
                               int z1=S.nextInt();
                               B[3].ChargeVehicle(z1,M.F[FloorNumber].getCostperUnitCharge());
                            }
                        }
                        int slot = M.F[FloorNumber].AllocateParkingLot(B[ChoiceOfVehicle-1]);// function for alloactinf slot and getting Slot Id
                        int xcordinate = --slot/(M.F[FloorNumber].s[0].length);
                        int ycordinate =slot%(M.F[FloorNumber].s[0].length);
                        M.F[FloorNumber].s[xcordinate][ycordinate].RemainingSpace-=1;// updating space occupied
                        if(M.F[FloorNumber].s[xcordinate][ycordinate].RemainingSpace==0)// makig boolean false
                        {
                            M.F[FloorNumber].s[xcordinate][ycordinate].IsSlotEmpty=false;
                        }
                        M.F[FloorNumber].en[x3].IsEmpty= true;
                        M.ListVehicles.add(B[ChoiceOfVehicle-1]);// adding vehicles to list 
                    }
        else{System.out.println("Floor is full choose again");}
 //       S.close();
    }

    protected void Exit(Management M,int ChoiceOfVehicle)
    {
        Scanner S11 = new Scanner(System.in);
        System.out.println(" Enter The Floor you Parked Your Vehicle");
                    int x2 = S11.nextInt()-1;
                    while(x2>M.F.length || x2<0)
                    {
                        System.out.println("Enter Floor again (you entered invalid Floor)");
                        x2 = S11.nextInt()-1;
                    }
                    int slot = 0,VehicleIdInList= -1;
                    boolean VehicleIsPresent=true;
                    while(VehicleIdInList<0 && VehicleIsPresent)
                    {
                        System.out.println("Enter Your Your Full Name as registerd in Info");
                        String Name = S11.next();
                        for(int i=0;i<M.ListVehicles.size();i++)
                        {
                            if(M.ListVehicles.get(i).OwnerName.compareTo(Name)==0)
                            {
                                slot = M.ListVehicles.get(i).AllocatedParkingLot;
                                VehicleIdInList= i;
                                break;
                            }
                        }
                        if(VehicleIdInList==-1)
                        {
                            System.out.println("Vehicle not found , Enter name Correctly");
                            System.out.println("If Your Vehicle is not parked ,Enter -2 ");
                        }
                        if(VehicleIdInList==-2)
                        {
                            VehicleIsPresent=false;
                        }
                    }
                    if(VehicleIdInList>=0)
                    {
                        System.out.println("Choose Exit");
                        M.F[x2].DisplayExits();
                        int x3 = S11.nextInt()-1;
                        M.F[x2].ex[x3].IsEmpty=false;
                        M.F[x2].ex[x3].AssignOutTime(M.ListVehicles.get(VehicleIdInList));
                        slot = M.ListVehicles.get(VehicleIdInList).AllocatedParkingLot;
                        int xcordinate = --slot/(M.F[x2].s[0].length); // calculating array indicies from slot
                        int ycordinate =slot%(M.F[x2].s[0].length);
                        M.F[x2].s[xcordinate][ycordinate].MakeSlotFree(1);
                        M.F[x2].ex[x3].CheckPayment(M.ListVehicles.get(VehicleIdInList),M.F[x2].cost);// checking and paying bill
                        M.F[x2].ex[x3].PaymentOptions();
                        int x4 =S11.nextInt();
                        while(x4!=1&& x4!=2)
                        {
                            System.out.println("Invalid Payment Method,Enter Choice Again");
                            x4=S11.nextInt();
                        }
                        if(x4==1)
                        {
                            M.ListVehicles.get(VehicleIdInList).PaymentMethod="Card";
                        }
                        else if(x4==2)
                        {
                            M.ListVehicles.get(VehicleIdInList).PaymentMethod="Attendant";
                        }
                        M.ListVehicles.get(VehicleIdInList).PayFee();
                        M.F[x2].ex[x3].CheckPayment(M.ListVehicles.get(VehicleIdInList),M.F[x2].cost);
                        M.F[x2].ex[x3].IsEmpty=true;
                    }
   //                 S11.close();
    }
}

