package caseStudyV1_1;

public class Exit extends Entries_Exits
{
    Exit(int id, boolean IsEmpty)
    {
        super(id, IsEmpty);
    }
    protected void CheckPayment(Vehicle v, int cost)
    {
        if(v.IsFeePayed)
        {
            return;
        }
        else
        {
            System.out.println("Generating Bill");
            long TotalBill=this.GenerateBill(((v.OutTime)-(v.InTime))/1000,cost );
            long BillToPayed = TotalBill-v.BillPayed;
            System.out.println("Your Bill is "+ BillToPayed+" Including advance Payed");
            v.BillPayed=TotalBill;
            return;
        }
    }
    public void AssignOutTime(Vehicle v)
    {
        v.OutTime = System.currentTimeMillis();
    }
    @Override
    public void PaymentOptions()
    {
        System.out.println(" 1 To Card ");
        System.out.println(" 2 To attendant ");
    }
    public long GenerateBill(long time,int cost)
    {
        long Bill=0;
        System.out.println(time);
        Bill += time*cost;
        return Bill;
    }

}
