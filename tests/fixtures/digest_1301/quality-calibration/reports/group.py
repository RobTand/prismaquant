print(json.dumps({'status': 'passed', 'policy': binding, 'served_quantizer': identity,
                  'old_maxima_unchanged': True, 'a8_qdq_and_identity_unchanged': True,
                  'groups': observations}, sort_keys=True))
