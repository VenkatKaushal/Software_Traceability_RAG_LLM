package caseStudyV1_1;

 abstract class Entries_Exits
{
    int id;
    boolean IsEmpty;
    abstract void CheckPayment(Vehicle v,int cost);
    Entries_Exits(int id ,boolean IsEmpty)
    {
        this.id = id;
        this.IsEmpty = IsEmpty;
    }
    abstract void PaymentOptions();
}

