package caseStudyV1_1;

public class Entry extends Entries_Exits
{
    Entry(int id, boolean IsEmpty)
    {
        super(id, IsEmpty);
    }
    public void CheckPayment(Vehicle v,int cost)
    {
        if(v.IsFeePayed)
        {
            v.IsFeePayed=false;
        }
        else
        {
            return;
        }
    }
    @Override
    public void PaymentOptions()
    {
        System.out.println(" 1 To Card ");
        System.out.println(" 2 To attendant ");
        System.out.println(" 3 To Portal ");
        System.out.println(" -1 at exit ");
    }
   public void PrePayment(long Advance,Vehicle V)
    {
        V.BillPayed=Advance;
    }
    public void AssignInTime(Vehicle v)
    {
        v.InTime = System.currentTimeMillis();
    }
}
