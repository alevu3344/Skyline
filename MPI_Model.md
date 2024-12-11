Our model is that of a distributed-memory cluster; that
is, if n is the problem size, our computational environ-
ment consists of p independent processors (workstations)
P0, P1, . . . , Pp−1, each with O(n/p) main memory, con-
nected through an interconnection network. Communica-
tion between the processors happens through the explicit
exchange of messages.
As the divide-and-conquer algorithm of [10], our algo-
rithm (see Algorithm 1) exploits that, for any decomposi-
tion of the point set S into subsets S0, S1, . . . , Sp−1, we
have sky(S) = sky(sky(S0) ∪ sky(S1) ∪ · · · ∪ sky(Sp−1)).
Thus, we can assign a subset Si of n/p points to each pro-
cessor Pi. In the first round, each processor Pi locally com-
putes sky(Si). The second round then computes sky(S)
from the local skylines computed by the processors.
To avoid that some processor computes a big local sky-
line all of whose member points are dominated by points
assigned to some other processor, our goal is to ensure that
each set Si is a sample of S whose structure is similar to
S. To this end, we distribute the points randomly over the
processors. After this distribution step, the first round of the
algorithm is straightforward: each processor locally applies
a sequential skyline algorithm to its point set.
The second round requires more care. Let S′ be the
union of the local skylines computed by all processors. If
|S′| ≤ n/p, we use an all-to-all communication to send S′
to every processor. Each processor Pi then determines for a
subset S′
i ⊆ S′ of points which points in S′
i are dominated
by points in S′ and removes these points from S′
i. At the
end of this round, we perform another all-to-all communi-
cation to collect the points in sets S′
i that were not deleted in
processor P0. These points form sky(S), and processor P0
returns this set or a sample L ⊆ sky(S) of size k, whichever
is smaller.
If |S′| > n/p, we pick a set S′′ of n/p random points
from S′ and send S′′ to every processor. Each processor Pi
now determines which points in S′′ are dominated by points
in sky(Si) and marks them. The marked points are then
sent back to the processors whence they came, and every
processor declares the points it contributed to S′′ and which
are not sent back to it by any other processor to be members
of sky(S). We iterate this process until we have processed
all points in S′ or we have identified at least k members
of sky(S). The fact that we choose each sample of size
n/p uniformly at random from S′ ensures that, if we do
not output all of sky(S), the sample we output is uniformly
distributed and is representative of the points in sky(S).
This general framework in principle allows the use of
any sequential skyline algorithm to perform the local sky-
line computations in the first round, as well as the filtering
step in the second round. 