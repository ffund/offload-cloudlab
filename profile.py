
"""Four nodes in a line with a server type c240g5 (One NVIDIA 12GB PCI P100 GPU)

Instructions:
Use the Wisconsin cluster to instantiate this profile.
"""

#
# NOTE: This code was machine converted. An actual human would not
#       write code like this!
#

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
# Import the Emulab specific extensions.
import geni.rspec.emulab as emulab

# Create a portal object,
pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

# Node client
node_client = request.RawPC('client')
node_client.routable_control_ip = True
node_client.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node_client.Site('Site 1')
iface0 = node_client.addInterface('interface-0', pg.IPv4Address('10.10.1.100','255.255.255.0'))
bs_client = node_client.Blockstore("bs1", "/data")
bs_client.size = "30GB"

# Node rtr1
node_rtr1 = request.RawPC('rtr1')
node_rtr1.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node_rtr1.Site('Site 1')
iface1 = node_rtr1.addInterface('interface-1', pg.IPv4Address('10.10.1.1','255.255.255.0'))
iface2 = node_rtr1.addInterface('interface-2', pg.IPv4Address('10.10.2.1','255.255.255.0'))

# Node rtr2
node_rtr2 = request.RawPC('rtr2')
node_rtr2.disk_image = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD'
node_rtr2.Site('Site 1')
iface3 = node_rtr2.addInterface('interface-3', pg.IPv4Address('10.10.2.2','255.255.255.0'))
iface4 = node_rtr2.addInterface('interface-4', pg.IPv4Address('10.10.3.2','255.255.255.0'))

# Node server
node_server = request.RawPC('server')
node_server.routable_control_ip = True
node_server.hardware_type = 'c240g5'
node_server.disk_image = 'urn:publicid:IDN+wisc.cloudlab.us+image+nyunetworks-PG0:offload.server:0'
node_server.Site('Site 1')
iface5 = node_server.addInterface('interface-5', pg.IPv4Address('10.10.3.100','255.255.255.0'))
bs_server = node_server.Blockstore("bs2", "/data")
bs_server.size = "100GB"

# Link link-0
link_0 = request.Link('link-0')
link_0.Site('undefined')
link_0.addInterface(iface0)
link_0.addInterface(iface1)

# Link link-1
link_1 = request.Link('link-1')
link_1.Site('undefined')
link_1.addInterface(iface2)
link_1.addInterface(iface3)

# Link link-2
link_2 = request.Link('link-2')
link_2.Site('undefined')
link_2.addInterface(iface4)
link_2.addInterface(iface5)


# Print the generated rspec
pc.printRequestRSpec(request)

