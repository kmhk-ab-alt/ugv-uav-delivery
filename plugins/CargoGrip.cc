// Compliant virtual gripper/winch. Applies equal/opposite forces to payload and carrier.
// This is a simplified end effector, not an articulated arm or flexible cable model.
#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <mutex>
#include <sstream>
#include <map>
#include <vector>
#include <cmath>
namespace pai {
class CargoGrip: public gz::sim::System, public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate {
 struct Grip{gz::sim::Entity parent,child;double length,mass;bool ground;gz::sim::Entity second=0;};
 gz::transport::Node node;std::mutex mutex;std::vector<std::string> queue;std::map<std::string,Grip> grips;
 public:void Configure(const gz::sim::Entity&,const std::shared_ptr<const sdf::Element>&,gz::sim::EntityComponentManager&,gz::sim::EventManager&)override{
  node.Subscribe("/pai/cargo/command",&CargoGrip::Command,this);
 }
 void Command(const gz::msgs::StringMsg &msg){std::lock_guard<std::mutex> lock(mutex);queue.push_back(msg.data());}
 void PreUpdate(const gz::sim::UpdateInfo &info,gz::sim::EntityComponentManager &ecm)override{
  if(info.paused)return;
  std::vector<std::string> commands;{std::lock_guard<std::mutex> lock(mutex);commands.swap(queue);}
  for(const auto &text:commands){
   std::istringstream s(text);std::string op,pkg,owner;double length=.3,mass=.6;s>>op>>pkg;
   if(op=="detach"){grips.erase(pkg);continue;}
   if(op!="attach"||!(s>>owner>>length>>mass)||!std::isfinite(length)||!std::isfinite(mass)||length<0||length>4||mass<=0||mass>5)continue;
   std::string secondary;auto plus=owner.find('+');if(plus!=std::string::npos){secondary=owner.substr(plus+1);owner=owner.substr(0,plus);}
   auto p=ecm.EntityByComponents(gz::sim::components::Model(),gz::sim::components::Name(owner));
   auto c=ecm.EntityByComponents(gz::sim::components::Model(),gz::sim::components::Name(pkg));
   if(!p||!c)continue;
   auto pl=gz::sim::Model(p).LinkByName(ecm,"body"),cl=gz::sim::Model(c).LinkByName(ecm,"body");
   if(!pl||!cl)continue;
   gz::sim::Link(pl).EnableVelocityChecks(ecm);gz::sim::Link(cl).EnableVelocityChecks(ecm);
   gz::sim::Entity extra=0;
   if(!secondary.empty()){
    auto em=ecm.EntityByComponents(gz::sim::components::Model(),gz::sim::components::Name(secondary));
    if(em)extra=gz::sim::Model(em).LinkByName(ecm,"body");
    if(extra)gz::sim::Link(extra).EnableVelocityChecks(ecm);
   }
   grips[pkg]={pl,cl,length,mass,owner.rfind("ground_",0)==0,extra};
  }
  for(const auto &entry:grips){
   const auto &g=entry.second;
   std::vector<gz::sim::Entity> parents{g.parent};if(g.second)parents.push_back(g.second);
   for(size_t i=0;i<parents.size();++i){
    gz::sim::Link parent(parents[i]),child(g.child);
    auto pp=gz::sim::worldPose(parents[i],ecm),cp=gz::sim::worldPose(g.child,ecm);
    gz::math::Vector3d offset=(i==0&&g.ground)?gz::math::Vector3d(.45,0,.28):gz::math::Vector3d(0,0,-g.length);
    auto target=pp.Pos()+pp.Rot().RotateVector(offset);
    auto pv=parent.WorldLinearVelocity(ecm).value_or(gz::math::Vector3d::Zero);
    auto cv=child.WorldLinearVelocity(ecm).value_or(gz::math::Vector3d::Zero);
    auto force=((target-cp.Pos())*90+(pv-cv)*12+gz::math::Vector3d(0,0,g.mass*9.81))/parents.size();
    if(force.Length()>45)force*=45/force.Length();
    child.AddWorldForce(ecm,force);parent.AddWorldForce(ecm,-force,(i==0&&g.ground)?offset:gz::math::Vector3d(0,0,-.1));
   }
  }
 }
};}
GZ_ADD_PLUGIN(pai::CargoGrip,gz::sim::System,pai::CargoGrip::ISystemConfigure,pai::CargoGrip::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(pai::CargoGrip,"pai::CargoGrip")
