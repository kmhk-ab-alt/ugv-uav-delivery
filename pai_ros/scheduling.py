"""Dispatch policy boundary: learned policies can replace ranking, not safety."""
class PriorityDeadlineScheduler:
    name='priority_then_deadline'
    def rank(self, jobs, now):
        # No ground-truth visual labels are used to rank deliveries.
        return sorted(jobs,key=lambda j:(-j.priority,j.deadline,j.name))


def parse_priorities(value, count):
    if not value or not value.strip(): return None
    try: priorities=[int(x.strip()) for x in value.split(',')]
    except ValueError: raise ValueError('priorities must be comma-separated integers 1..3') from None
    if len(priorities)!=count or any(p not in (1,2,3) for p in priorities):
        raise ValueError('provide one priority 1..3 for each package')
    return priorities
