output "floating_ip_out" {
  description = "Floating IP assigned to node1"
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

output "volume_id" {
  value = data.openstack_blockstorage_volume_v3.existing_volume.id
}

output "volume_size" {
  value = data.openstack_blockstorage_volume_v3.existing_volume.size
}

output "volume_status" {
  value = data.openstack_blockstorage_volume_v3.existing_volume.status
}

